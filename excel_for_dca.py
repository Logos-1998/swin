import os
import torch
import argparse
import pandas as pd
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader

# 复用项目模块
from config import get_config
from models import build_model
from data.build import build_transform

# 引入您项目可能使用的临床融合 Wrapper (若未使用融合模块，会自动跳过)
try:
    from data.build import ClinicalDatasetWrapper
except ImportError:
    pass

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
WEIGHT_PATH = r"D:\Documents\Swin-Transformer\output\exp4_full\checkpoint_best.pth"

# 2. 配置文件路径
CONFIG_PATH = r"D:\Documents\Swin-Transformer\configs\exp3_fusion.yaml"

# 3. 测试集路径 (文件夹下直接是类别子文件夹，例如 dataset/test/Normal)
TEST_DATA_PATH = r"D:\Documents\Swin-Transformer\dataset\test"

# 4. 临床特征 CSV 路径 (如配置文件中未开启 FUSION，可忽略此项)
CLINICAL_CSV = r"D:\Documents\Swin-Transformer\dataset\clinical_data.csv"

# 5. 结果保存目录及文件命名
OUTPUT_ROOT = r"D:\Documents\Swin-Transformer\results"
SUB_FOLDER_NAME = "exp4_full_evaluation"
EXCEL_FILENAME = "test_predictions.xlsx"

# ==============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cfg', type=str, default=CONFIG_PATH, help='path to config file')
    parser.add_argument("--opts", help="Modify config options", default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, default=64, help="batch size")
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

    # --- 3. 手动构建测试集加载器 ---
    print(f"⏳ Loading test data directly from: {TEST_DATA_PATH}")

    transforms = build_transform(is_train=False, config=config)

    # (A) 基础数据集加载
    try:
        base_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transforms)
    except FileNotFoundError:
        print(f"❌ Error: 路径 {TEST_DATA_PATH} 不存在或不包含图片类别文件夹。")
        return

    # (B) 获取类别信息与图片路径
    class_names = base_dataset.classes
    class_to_idx = base_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"🏷️ Classes found ({len(class_names)}): {class_to_idx}")

    image_paths = [s[0] for s in base_dataset.samples]
    image_names = [os.path.basename(p) for p in image_paths]

    # (C) 处理临床多模态融合 (如果配置文件中开启了 FUSION)
    dataset_val = base_dataset
    if config.MODEL.FUSION.ENABLED:
        print(f"🧬 检测到融合模块已开启，正在加载临床数据: {CLINICAL_CSV}")
        dataset_val = ClinicalDatasetWrapper(
            base_dataset,
            csv_path=CLINICAL_CSV,
            clinical_dim=config.MODEL.FUSION.CLINICAL_DIM
        )

    # (D) 构建 DataLoader (必须 shuffle=False 以保证输出结果与文件名对齐)
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
    # 注意：此处依赖您的 engine.py 中的 validate 函数返回包含 targets, preds, probs, acc 的字典
    final_stats = validate(config, data_loader_val, model, class_names=class_names)

    targets = np.array(final_stats['targets'])
    preds = np.array(final_stats['preds'])
    probs = np.array(final_stats['probs'])

    # --- 7. 导出预测结果到 Excel ---
    print("📝 正在生成标准格式的预测结果 Excel 表格...")

    # 🔴 核心安全检查：强制映射关系 🔴
    expected_mapping = {0: 'Normal', 1: 'OP', 2: 'OPA'}

    for expected_idx, expected_cls in expected_mapping.items():
        if expected_cls not in class_to_idx or class_to_idx[expected_cls] != expected_idx:
            print(f"❌ [致命警告] 类别映射不一致！")
            print(f"   期望映射: {expected_mapping}")
            print(f"   实际映射: {class_to_idx}")
            print("   这会导致您后续画 DCA 时类别错乱！请检查测试集文件夹的字母排序！")

    # 动态获取对应类别的概率列，若因故缺失则填入 NaN
    prob_normal = probs[:, class_to_idx.get('Normal', 0)] if 'Normal' in class_to_idx else np.nan
    prob_op     = probs[:, class_to_idx.get('OP', 1)] if 'OP' in class_to_idx else np.nan
    prob_opa    = probs[:, class_to_idx.get('OPA', 2)] if 'OPA' in class_to_idx else np.nan

    df_dict = {
        'Image_Name': image_names,
        'True_Class': [idx_to_class.get(t, f"Unknown_{t}") for t in targets],
        'Pred_Class': [idx_to_class.get(p, f"Unknown_{p}") for p in preds],
        'Prob_Normal': prob_normal,
        'Prob_OP': prob_op,
        'Prob_OPA': prob_opa
    }

    # 判断正确性直接比对 numpy 数组
    df_dict['Is_Correct'] = (targets == preds).astype(bool)

    df = pd.DataFrame(df_dict)
    excel_path = os.path.join(config.OUTPUT, EXCEL_FILENAME)
    df.to_excel(excel_path, index=False)
    print(f"✅ Excel 已成功保存至: {excel_path}")

    # --- 8. 绘图与报告 ---
    print("📊 Generating metrics plots...")
    try:
        plot_confusion_matrix(targets, preds, class_names, config.OUTPUT)
        plot_roc_curve(targets, probs, class_names, config.OUTPUT)
        generate_classification_report(targets, preds, class_names, config.OUTPUT)
    except Exception as e:
        print(f"⚠️ Error during plotting: {e}")

    print(f"\n🎉 Done! Top-1 Acc: {final_stats['acc']:.4f}%")

if __name__ == '__main__':
    main()