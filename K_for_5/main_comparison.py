import os
import sys
import time
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torch.backends.cudnn as cudnn

# 引入项目定义的模块 (完美复用原项目生态)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config
from data import build_loader
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from evaluate import train_one_epoch, validate  # 注意：你的文件叫 evaluate.py 还是 engine.py，请保持一致
from utils import (
    load_checkpoint, save_checkpoint, find_unique_output_dir,
    plot_loss_curve, plot_confusion_matrix, plot_roc_curve, generate_classification_report,
    EarlyStopping
)

# 定义 Focal Loss
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
    parser = argparse.ArgumentParser('Torchvision Models Comparative Study', add_help=False)
    # 核心配置文件读取
    parser.add_argument('--cfg', type=str, default = r'D:\Documents\Swin-Transformer\configs\efficientnetv2.yaml', metavar="FILE", help='path to config file')
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

def build_tv_model(model_name, num_classes):
    """动态适配各个框架的分类层，并支持从头训练"""
    print(f"🏗️  Creating Torchvision Model: {model_name} (From Scratch)")
    if model_name == 'resnet50':
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = tv_models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'efficientnet_v2_s':
        model = tv_models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'vit_b_16':
        model = tv_models.vit_b_16(weights=None, image_size=256)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")
    return model

def main(config):
    if not torch.cuda.is_available():
        print("❌ Error: CUDA is not available.")
        return

    device = torch.device("cuda")
    print(f"🖥️  Device: {torch.cuda.get_device_name(0)}")

    # 1. 自动管理输出目录
    final_output_dir = find_unique_output_dir(config.OUTPUT, config.MODEL.NAME)
    os.makedirs(final_output_dir, exist_ok=True)

    config.defrost()
    config.OUTPUT = final_output_dir
    config.freeze()

    # 2. 随机种子设定
    seed = getattr(config, 'SEED', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 3. 完美复用原项目的数据流水线 (包含 Mixup 支持等)
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    if hasattr(dataset_train, 'dataset') and hasattr(dataset_train.dataset, 'classes'):
        class_names = dataset_train.dataset.classes
    elif hasattr(dataset_train, 'classes'):
        class_names = dataset_train.classes
    else:
        class_names = [str(i) for i in range(config.MODEL.NUM_CLASSES)]

    print(f"📦 Classes: {class_names}")

    # 4. 计算 Focal Loss 的平衡权重
    if hasattr(dataset_train, 'dataset') and hasattr(dataset_train.dataset, 'targets'):
        train_targets = dataset_train.dataset.targets
    elif hasattr(dataset_train, 'targets'):
        train_targets = dataset_train.targets
    else:
        train_targets = [t for _, t, _ in dataset_train]  # 兼容 VertebraDatasetWrapper 返回三元组

    class_counts = np.bincount(train_targets)
    weights = len(train_targets) / (len(class_counts) * class_counts + 1e-6)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # 5. 构建 Torchvision 对比模型
    model = build_tv_model(config.MODEL.NAME, config.MODEL.NUM_CLASSES)
    model.to(device)
    print(f"📊 Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M")

    # 6. 完美复用原项目的优化器与调度器 (彻底解决 step_update 报错)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # 7. 损失函数
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    print("🎯 Loss Mode: Focal Loss (Alpha-Weighted, Gamma=2.0)")

    # 8. 训练循环
    print("🚀 Start Training...")
    start_time = time.time()
    train_losses, val_losses = [], []
    max_accuracy = 0.0

    early_stopper = EarlyStopping(
        patience=config.TRAIN.EARLY_STOPPING.PATIENCE, verbose=True
    ) if config.TRAIN.EARLY_STOPPING.ENABLED else None

    is_middle_saved = False

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        # 调用原版 evaluate.py 里的训练代码（它自带 config 解析和 step_update）
        train_stats = train_one_epoch(
            config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, class_names
        )
        train_losses.append(train_stats['loss'])

        # 验证
        val_stats = validate(config, data_loader_val, model, class_names=class_names)
        val_losses.append(val_stats['loss'])
        acc1 = val_stats['acc']

        # 中期保存
        if getattr(config.TRAIN, 'MIDDLE_ACC', 0.0) > 0.0 and not is_middle_saved:
            if acc1 >= config.TRAIN.MIDDLE_ACC:
                print(f"\n✅ Reached Middle Target ({config.TRAIN.MIDDLE_ACC*100:.1f}%)! Saving checkpoint_middle.pth...")
                save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger=None, filename='checkpoint_middle')
                is_middle_saved = True

        # 最优保存
        is_best = acc1 > max_accuracy
        if is_best:
            max_accuracy = acc1
            print(f"🌟 New Best Accuracy: {max_accuracy * 100:.2f}%")

        save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, is_best=is_best, logger=None)

        print(f"Epoch [{epoch}] | Train Loss: {train_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f} | Val Acc: {acc1 * 100:.2f}%")

        # 难样本完美终止逻辑
        if getattr(config.TRAIN, 'TARGET_ACC', 0.0) > 0.0 and acc1 >= config.TRAIN.TARGET_ACC:
            target_min_f1 = getattr(config.TRAIN, 'TARGET_MIN_F1', 0.0)
            if target_min_f1 > 0.0 and 'f1_per_class' in val_stats:
                f1_scores = val_stats['f1_per_class']
                op_idx = class_names.index('OP') if 'OP' in class_names else -1
                opa_idx = class_names.index('OPA') if 'OPA' in class_names else -1

                op_f1 = f1_scores[op_idx] if op_idx != -1 else 1.0
                opa_f1 = f1_scores[opa_idx] if opa_idx != -1 else 1.0

                if (op_idx != -1 and op_f1 < target_min_f1) or (opa_idx != -1 and opa_f1 < target_min_f1):
                    print(f"\n⚠️ 总体ACC达标 ({acc1*100:.2f}%)，但OP_F1({op_f1:.4f})或OPA_F1({opa_f1:.4f}) 未达标，继续挖掘难样本...")
                    continue
            print(f"\n🎉 总体ACC与各项核心F1均达标，完美停止!!!")
            break

        # 早停
        if early_stopper:
            early_stopper(acc1, val_stats['loss'])
            if early_stopper.early_stop:
                print(f"🛑 Early stopping triggered at epoch {epoch}!")
                break

    # 9. 生成报告
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"🏁 Finished. Total time: {total_time_str}")

    plot_loss_curve(train_losses, val_losses, final_output_dir)

    best_path = os.path.join(final_output_dir, 'checkpoint_best.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])

        final_stats = validate(config, data_loader_val, model, class_names=class_names)
        try:
            plot_confusion_matrix(final_stats['targets'], final_stats['preds'], class_names, final_output_dir)
            plot_roc_curve(final_stats['targets'], final_stats['probs'], class_names, final_output_dir)
            generate_classification_report(final_stats['targets'], final_stats['preds'], class_names, final_output_dir)
            print(f"📂 Artifacts saved to: {final_output_dir}")
        except Exception as e:
            print(f"❌ Error in reporting: {e}")

if __name__ == '__main__':
    args, config = parse_option()
    main(config)