import os
import torch
import argparse
from torchvision import datasets
from torch.utils.data import DataLoader

# 复用项目模块
from config import get_config
from models import build_model
from data.build import build_transform  # 只复用 transform 构建逻辑
from engine import validate
from utils import (
    plot_confusion_matrix,
    plot_roc_curve,
    generate_classification_report
)

# ==============================================================================
# 🛠️ 用户配置区
# ==============================================================================

# 1. 权重路径
WEIGHT_PATH = r"D:\Documents\Swin-Transformer\output\exp1_baseline_1\checkpoint_best.pth"

# 2. 配置文件路径
CONFIG_PATH = r"D:\Documents\Swin-Transformer\configs\exp1_baseline.yaml"

# 3. 测试集路径 (文件夹下直接是类别子文件夹，例如 dataset/test/0_normal)
TEST_DATA_PATH = r"D:\Documents\Swin-Transformer\dataset\test"

# 4. 结果保存目录
OUTPUT_ROOT = r"D:\Documents\Swin-Transformer\results"
SUB_FOLDER_NAME = "baseline"

# ==============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cfg', type=str, default=CONFIG_PATH, help='path to config file')
    parser.add_argument("--opts", help="Modify config options", default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, default=64, help="batch size") # 默认 Batch Size
    parser.add_argument('--data-path', type=str, default=TEST_DATA_PATH, help='path to dataset')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O1')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--output', default=OUTPUT_ROOT, type=str, help='root output folder')
    return parser

def main():
    # --- 1. 初始化配置 ---
    args, _ = get_args_parser().parse_known_args()
    config = get_config(args)

    config.defrost()
    config.DATA.DATA_PATH = TEST_DATA_PATH
    config.MODEL.RESUME = WEIGHT_PATH
    config.EVAL_MODE = True
    config.OUTPUT = os.path.join(OUTPUT_ROOT, SUB_FOLDER_NAME)
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    print(f"📂 评估结果将保存至: {config.OUTPUT}")

    # --- 2. 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- 3. 手动构建测试集加载器 (绕过 build_loader) ---
    print(f"⏳ Loading test data directly from: {TEST_DATA_PATH}")

    # (A) 获取验证/测试专用的 Transform (不做增强，只做 Resize/Normalize)
    transforms = build_transform(is_train=False, config=config)

    # (B) 直接使用 ImageFolder 读取指定路径，不再自动拼接 'val' 或 'train'
    try:
        dataset_val = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transforms)
    except FileNotFoundError:
        print(f"❌ Error: 路径 {TEST_DATA_PATH} 不存在或不包含图片类别文件夹。")
        return

    # (C) 获取类别信息
    class_names = dataset_val.classes
    print(f"🏷️ Classes found ({len(class_names)}): {class_names}")

    # (D) 构建 DataLoader
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # --- 4. 构建模型 ---
    print(f"🚀 Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    # 临时解冻以更新类别数，防止 config 和数据集不一致
    config.defrost()
    config.MODEL.NUM_CLASSES = len(class_names)
    config.freeze()

    model = build_model(config)
    model.to(device)

    # --- 5. 加载权重 ---
    print(f"📥 Loading checkpoint: {WEIGHT_PATH}")
    if not os.path.isfile(WEIGHT_PATH):
        print("❌ Checkpoint file not found!")
        return

    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # --- 6. 推理与评估 ---
    print("🔥 Starting inference...")
    final_stats = validate(config, data_loader_val, model, class_names=class_names)

    # --- 7. 绘图与报告 ---
    print("📊 Generating metrics...")
    targets = final_stats['targets']
    preds = final_stats['preds']
    probs = final_stats['probs']

    try:
        plot_confusion_matrix(targets, preds, class_names, config.OUTPUT)
        plot_roc_curve(targets, probs, class_names, config.OUTPUT)
        generate_classification_report(targets, preds, class_names, config.OUTPUT)
    except Exception as e:
        print(f"⚠️ Error during plotting: {e}")

    print(f"\n✅ Done! Top-1 Acc: {final_stats['acc']:.4f}%")

if __name__ == '__main__':
    main()