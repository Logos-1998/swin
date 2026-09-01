import os
import sys
import time
import random
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# 将父目录加入系统路径，以便在 K_for_5 文件夹中能够无缝导入原项目的底层模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from models import build_model
from data import build_loader
from optimizer import build_optimizer
from lr_scheduler import build_scheduler

# 【关键修改1】: 将训练和验证引擎的导入路径改为即将新建的 evaluate.py
from evaluate import train_one_epoch, validate

from utils import (
    load_checkpoint, save_checkpoint, auto_resume_helper, find_unique_output_dir,
    plot_loss_curve, plot_confusion_matrix, plot_roc_curve, generate_classification_report,
    EarlyStopping
)
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer V2 Training Script (K-Fold Adapted)', add_help=False)

    # 1. 路径与配置参数
    parser.add_argument('--cfg', type=str, default=r'D:\Documents\Swin-Transformer\configs\exp2_esc.yaml', metavar="FILE", help='path to config file')
    parser.add_argument("--opts", help="Modify config options", default=None, nargs='+')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root output folder')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--tag', help='tag of experiment')

    # 2. 基础运行参数
    parser.add_argument('--batch-size', type=int, help="batch size")
    parser.add_argument('--zip', action='store_true', help='use zipped dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'])
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # 3. 调试超参
    parser.add_argument('--epochs', type=int, help="number of epochs")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--weight-decay', type=float, help="weight decay")
    parser.add_argument('--early-stopping', action='store_true', help='enable early stopping')
    parser.add_argument('--patience', type=int, help='patience')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

def main(config):
    # --- 1. 环境初始化 ---
    if not torch.cuda.is_available():
        print("❌ Error: CUDA is not available.")
        return

    device = torch.device("cuda")
    print(f"✅ Device: {torch.cuda.get_device_name(0)}")

    # 自动管理输出目录
    final_output_dir = find_unique_output_dir(config.OUTPUT, config.MODEL.NAME)
    os.makedirs(final_output_dir, exist_ok=True)

    config.defrost()
    config.OUTPUT = final_output_dir
    config.freeze()

    # 设置随机种子
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # --- 2. 数据加载 ---
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    # 获取类别名
    if hasattr(dataset_train, 'dataset') and hasattr(dataset_train.dataset, 'classes'):
        class_names = dataset_train.dataset.classes
    elif hasattr(dataset_train, 'classes'):
        class_names = dataset_train.classes
    else:
        class_names = [str(i) for i in range(config.MODEL.NUM_CLASSES)]

    print(f"📂 Classes: {class_names}")

    # --- 3. 类别权重计算 ---
    if hasattr(dataset_train, 'dataset') and hasattr(dataset_train.dataset, 'targets'):
        train_targets = dataset_train.dataset.targets
    elif hasattr(dataset_train, 'targets'):
        train_targets = dataset_train.targets
    else:
        # 【关键修改2】: 兼容修改后的 Dataset 返回的 (sample, target, vertebra_id) 三元组解析
        train_targets = [t for _, t, _ in dataset_train]

    class_counts = np.bincount(train_targets)
    weights = len(train_targets) / (len(class_counts) * class_counts + 1e-6)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # --- 4. 构建模型 ---
    print(f"🚀 Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.to(device)
    print(f"📊 Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    # --- 5. 优化器与调度器 ---
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # --- 6. 损失函数 ---
    # --- 6. 损失函数 (Focal Loss 解决不平衡与难样本) ---
    use_focal_loss = False
    if use_focal_loss:
        criterion = FocalLoss(weight=class_weights, gamma=2.0)
        print("⚖️ Loss Mode: Focal Loss (Alpha-Weighted, Gamma=2.0)")
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("⚖️ Loss Mode: Weighted CrossEntropy")

    # --- 7. 检查点恢复 ---
    max_accuracy = 0.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger=None)
        if config.EVAL_MODE:
            final_stats = validate(config, data_loader_val, model, class_names=class_names)
            print(f"Final Eval Accuracy (Vertebra-Level): {final_stats['acc'] * 100:.2f}%")
            return

    # --- 8. 训练循环 ---
    print("🔥 Start Training...")
    start_time = time.time()
    train_losses, val_losses = [], []

    early_stopper = EarlyStopping(
        patience=config.TRAIN.EARLY_STOPPING.PATIENCE,
        verbose=True
    ) if config.TRAIN.EARLY_STOPPING.ENABLED else None

    is_middle_saved = False

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        # 【关键修改3】: 将 mixup_fn 强制替换为 None，彻底关闭 Mixup，避免后续传入引擎时触发拆包异常
        train_stats = train_one_epoch(
            config, model, criterion, data_loader_train, optimizer, epoch, None, lr_scheduler,
            class_names=class_names
        )
        train_losses.append(train_stats['loss'])

        # 验证 Epoch (此处的 validate 将输出椎体级的聚合成级)
        val_stats = validate(config, data_loader_val, model, class_names=class_names)
        val_losses.append(val_stats['loss'])
        acc1 = val_stats['acc']

        if config.TRAIN.MIDDLE_ACC > 0.0 and not is_middle_saved:
            if acc1 >= config.TRAIN.MIDDLE_ACC:
                print(f"\n🚩 Reached Middle Target ({config.TRAIN.MIDDLE_ACC*100:.1f}%)! Saving checkpoint_middle.pth...")
                save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler,
                                is_best=False, logger=None, filename='checkpoint_middle')
                is_middle_saved = True

        # 保存策略
        is_best = acc1 > max_accuracy
        if is_best:
            max_accuracy = acc1
            print(f"⭐ New Best Accuracy (Vertebra-Level): {max_accuracy * 100:.2f}%")

        save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, is_best=is_best, logger=None)

        print(f"Epoch {epoch} | Train Loss: {train_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f} | Val Acc: {acc1 * 100:.2f}%")

        if config.TRAIN.TARGET_ACC > 0.0 and acc1 >= config.TRAIN.TARGET_ACC:
            min_f1 = getattr(config.TRAIN, 'TARGET_MIN_F1', 0.0)

            if min_f1 > 0.0 and 'f1_per_class' in val_stats:
                f1_scores = val_stats['f1_per_class']

                # 动态获取目标类别的索引（防止类别顺序变动）
                op_idx = class_names.index('OP') if 'OP' in class_names else -1
                opa_idx = class_names.index('OPA') if 'OPA' in class_names else -1

                op_f1 = f1_scores[op_idx] if op_idx != -1 else 1.0
                opa_f1 = f1_scores[opa_idx] if opa_idx != -1 else 1.0

                # 如果有任何一个核心类的 F1 未达标，则跳过中断，继续训练
                if (op_idx != -1 and op_f1 < min_f1) or (opa_idx != -1 and opa_f1 < min_f1):
                    print(f"\n⚠️ 总体 ACC 达标 ({acc1*100:.2f}%)，但 OP_F1({op_f1:.4f}) 或 OPA_F1({opa_f1:.4f}) 未达 {min_f1} 阈值，继续挖掘难样本...")
                    continue

            print(f"\n🎯 总体 ACC 与各项核心 F1 均达标，完美停止!!!")
            break

        if early_stopper:
            early_stopper(acc1, val_stats['loss'])
            if early_stopper.early_stop:
                print(f"🛑 Early stopping triggered at epoch {epoch}!")
                break

    # --- 9. 训练完成，生成最终报告 ---
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'✅ Finished. Total time: {total_time_str}')

    plot_loss_curve(train_losses, val_losses, final_output_dir)

    # 使用最佳权重运行最终测试
    best_path = os.path.join(final_output_dir, 'checkpoint_best.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])

    # 最终的椎体级指标计算与画图
    final_stats = validate(config, data_loader_val, model, class_names=class_names)
    try:
        plot_confusion_matrix(final_stats['targets'], final_stats['preds'], class_names, final_output_dir)
        plot_roc_curve(final_stats['targets'], final_stats['probs'], class_names, final_output_dir)
        generate_classification_report(final_stats['targets'], final_stats['preds'], class_names, final_output_dir)
        print(f"🎨 Artifacts saved to: {final_output_dir}")
    except Exception as e:
        print(f"⚠️ Error in reporting: {e}")

if __name__ == '__main__':
    args, config = parse_option()
    main(config)