import os
import time
import random
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# 引入我们写好的各个模块
from config import get_config
from models import build_model
from data import build_loader
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from engine import train_one_epoch, validate
from utils import (
    load_checkpoint, save_checkpoint, auto_resume_helper, find_unique_output_dir,
    plot_loss_curve, plot_confusion_matrix, plot_roc_curve, generate_classification_report,
    EarlyStopping
)

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer V2 Training Script', add_help=False)

    # -----------------------------------------------------------
    # 1. 基础与路径参数
    # -----------------------------------------------------------
    parser.add_argument('--cfg', type=str, default = r'E:\WM\Swin-Transformer\configs\exp1_baseline.yaml', metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--tag', help='tag of experiment')

    # -----------------------------------------------------------
    # 2. config.py 必须依赖的参数 (修复 AttributeError 的关键)
    # -----------------------------------------------------------
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # -----------------------------------------------------------
    # 3. 方便调试的超参入口 (REQ 2 & 4)
    # -----------------------------------------------------------
    parser.add_argument('--epochs', type=int, help="number of epochs")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--weight-decay', type=float, help="weight decay")

    # 早停控制
    parser.add_argument('--early-stopping', action='store_true', help='enable early stopping')
    parser.add_argument('--patience', type=int, help='early stopping patience')

    # 随机种子
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config

def main(config):
    # 1. 环境初始化
    if not torch.cuda.is_available():
        print("❌ Error: CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")
    print(f"✅ Running on Single GPU: {torch.cuda.get_device_name(0)}")

    # REQ 5: 自动管理输出目录
    # 例如 output/swinv2_fusion -> output/swinv2_fusion_1
    # 注意：config.OUTPUT 是根目录，config.MODEL.NAME 是模型名
    final_output_dir = find_unique_output_dir(config.OUTPUT, config.MODEL.NAME)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"🚀 Output directory set to: {final_output_dir}")

    # 更新 config 里的 output 路径以供后续使用
    config.defrost()
    config.OUTPUT = final_output_dir
    config.freeze()

    # 设置种子，保证可复现性
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 2. 构建数据加载器
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    # 获取类别名 (用于画图)
    if hasattr(dataset_train, 'dataset') and hasattr(dataset_train.dataset, 'classes'):
        # 针对 ClinicalDatasetWrapper 包装的情况
        class_names = dataset_train.dataset.classes
    elif hasattr(dataset_train, 'classes'):
        # 针对普通 ImageFolder
        class_names = dataset_train.classes
    else:
        # 兜底
        class_names = [str(i) for i in range(config.MODEL.NUM_CLASSES)]

    print(f"📂 Classes: {class_names}")
    print(f"✅ Data loaded: Train={len(dataset_train)}, Val={len(dataset_val)}")

    print("⚖️ Calculating class weights for Weighted Loss...")

    # 1. 获取所有训练标签
    # 注意：dataset_train 可能是 Wrapper，需要剥离取出底层的 ImageFolder 的 targets
    if hasattr(dataset_train, 'dataset') and hasattr(dataset_train.dataset, 'targets'):
        # Wrapper 模式
        train_targets = dataset_train.dataset.targets
    elif hasattr(dataset_train, 'targets'):
        # 普通 ImageFolder 模式
        train_targets = dataset_train.targets
    else:
        # 兜底：如果实在拿不到，只能遍历一遍 (比较慢，但安全)
        print("   Warning: Iterating dataset to count classes (slow)...")
        train_targets = []
        for _, t in dataset_train:
            train_targets.append(t)

    # 2. 统计数量
    class_counts = np.bincount(train_targets)
    total_samples = len(train_targets)
    num_classes = len(class_counts)

    # 打印分布
    print(f"   Class Distribution: {dict(zip(class_names, class_counts))}")

    # 3. 计算权重: Weight = Total / (NumClasses * Count)
    # 数量越少，权重越大
    weights = total_samples / (num_classes * class_counts + 1e-6) # 加微小值防除零

    # 4. 转为 Tensor 并移动到 GPU
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"   Computed Weights: {class_weights.cpu().numpy()}")
    print("   -> Weighted Loss Enabled!")
    # -----------------------------------------------------------

    # 3. 构建模型
    print(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.to(device)

    # 打印参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters / 1e6:.2f} M")

    # 4. 构建优化器
    optimizer = build_optimizer(config, model)

    # 5. 构建 LR 调度器
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # 6. 检查点恢复 (Resume)
    max_accuracy = 0.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT) # 注意：新目录是空的，这里可能需要在父目录找，暂时简化处理
        if resume_file:
            if config.MODEL.RESUME:
                print(f"Warning: --resume {config.MODEL.RESUME} is ignored. Use auto-resume {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger=None)
        # 如果仅仅是评估模式
        if config.EVAL_MODE:
            final_stats = validate(config, data_loader_val, model)
            print(f"Eval Accuracy: {final_stats['acc']:.2f}%")
            return

    # 7. 开始训练循环
    print("🚀 Start training...")
    start_time = time.time()
    train_losses = []
    val_losses = []

    # REQ 4: 早停初始化
    # 只有当配置里开启且提供了 patience 时才启用
    use_early_stopping = config.TRAIN.EARLY_STOPPING.ENABLED
    early_stopper = EarlyStopping(
        patience=config.TRAIN.EARLY_STOPPING.PATIENCE,
        verbose=True
    ) if use_early_stopping else None

    if use_early_stopping:
        print(f"🛑 Early Stopping Enabled (Patience={config.TRAIN.EARLY_STOPPING.PATIENCE})")

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        # 训练一个 Epoch
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        train_stats = train_one_epoch(
            config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
            class_names=class_names
        )
        train_losses.append(train_stats['loss'])

        # 验证
        val_stats = validate(config, data_loader_val, model, class_names=class_names)
        val_losses.append(val_stats['loss'])
        acc1 = val_stats['acc']

        # REQ 3: 只保存最佳权重 (Best) 和 最新权重 (Last)
        is_best = acc1 > max_accuracy
        if is_best:
            max_accuracy = acc1
            # [修正] 乘以 100
            print(f"🔥 New best accuracy: {max_accuracy * 100:.2f}%")

        save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, is_best=is_best, logger=None)

        print(f"Epoch {epoch} Summary | Train Loss: {train_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f} | Val Acc: {acc1 * 100:.2f}%")

        # Check Early Stopping
        if early_stopper:
            early_stopper(acc1, val_stats['loss'])
            if early_stopper.early_stop:
                print(f"🛑 Early stopping triggered at epoch {epoch}!")
                break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'✅ Training finished. Total time {total_time_str}')

    # =========================================================
    # REQ 6: 训练后生成详细报告 (使用最佳权重)
    # =========================================================
    print("\n🎨 Generating Final Reports & Plots using BEST model...")

    # 1. 绘制 Loss 曲线
    plot_loss_curve(train_losses, val_losses, final_output_dir)

    # 2. 加载最佳权重
    best_path = os.path.join(final_output_dir, 'checkpoint_best.pth')
    if os.path.exists(best_path):
        # 这是一个临时的 checkpoint 加载，不需要恢复 optimizer
        checkpoint = torch.load(best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"✅ Loaded best checkpoint from {best_path} for final evaluation.")
    else:
        print("⚠️ Best checkpoint not found, using current model for evaluation.")

    # 3. 运行最终验证
    final_stats = validate(config, data_loader_val, model)

    # 4. 生成图表
    try:
        plot_confusion_matrix(final_stats['targets'], final_stats['preds'], class_names, final_output_dir)
        plot_roc_curve(final_stats['targets'], final_stats['probs'], class_names, final_output_dir)
        generate_classification_report(final_stats['targets'], final_stats['preds'], class_names, final_output_dir)
        print(f"🎉 All artifacts saved to: {final_output_dir}")
    except Exception as e:
        print(f"⚠️ Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    args, config = parse_option()
    main(config)