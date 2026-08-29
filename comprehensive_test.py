import os
import time
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# 复用项目模块
from config import get_config
from models import build_model
from data.build import build_transform

# 引入临床融合 Wrapper
try:
    from data.build import ClinicalDatasetWrapper
except ImportError:
    pass

from engine import validate

# ==============================================================================
# 🛠️ 用户配置区
# ==============================================================================

# 测试集路径 (文件夹下直接是类别子文件夹，例如 dataset/test/Normal)
TEST_DATA_PATH = r"D:\Documents\Swin-Transformer\dataset\test"
CLINICAL_CSV = r"D:\Documents\Swin-Transformer\dataset\clinical_data.csv"

# 输出日志文件路径
LOG_FILE_PATH = r"results/middle.log"

# 定义要测试的4个模型 (对应: 配置文件路径, 权重文件路径, 显示名称)
# 请仔细核对这里的 YAML 路径和 PTH 路径是否正确！
MODELS_TO_TEST = [
    {
        "name": "1. Baseline (SwinV2-T)",
        "cfg": r"D:\Documents\Swin-Transformer\configs\exp1_baseline.yaml",
        "weight": r"D:\Documents\Swin-Transformer\output\exp1_baseline_1\checkpoint_best.pth"
    },
    {
        "name": "2. ESC Only (SwinV2 + ESC)",
        "cfg": r"D:\Documents\Swin-Transformer\configs\exp2_esc.yaml",
        "weight": r"D:\Documents\Swin-Transformer\output\exp2_esc\checkpoint_middle.pth"
    },
    {
        "name": "3. Fusion Only (SwinV2 + H-CQA)",
        "cfg": r"D:\Documents\Swin-Transformer\configs\exp4_full.yaml",
        "weight": r"D:\Documents\Swin-Transformer\output\exp3_fusion\checkpoint_middle.pth"
    },
    {
        "name": "4. Full Model (SwinV2 + ESC + H-CQA)",
        "cfg": r"D:\Documents\Swin-Transformer\configs\exp3_fusion.yaml",
        "weight": r"D:\Documents\Swin-Transformer\output\exp4_full\checkpoint_middle.pth"
    }
]

# ==============================================================================

# 构造一个假的 args 对象以便 get_config 使用
class DummyArgs:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = None
        self.batch_size = 32
        self.data_path = TEST_DATA_PATH
        self.zip = False
        self.cache_mode = 'part'
        self.resume = None
        self.accumulation_steps = None
        self.use_checkpoint = False
        self.amp_opt_level = 'O1'
        self.local_rank = 0
        self.tag = None
        self.eval = True
        self.throughput = False
        self.output = "output"

def calculate_specificity(cm, class_idx):
    """根据混淆矩阵计算指定类别的特异度 (Specificity)"""
    tn = np.sum(cm) - np.sum(cm[class_idx, :]) - np.sum(cm[:, class_idx]) + cm[class_idx, class_idx]
    fp = np.sum(cm[:, class_idx]) - cm[class_idx, class_idx]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def write_log(msg, file_handle=None):
    """同时打印到控制台和写入日志文件"""
    print(msg)
    if file_handle is not None:
        file_handle.write(msg + "\n")

def evaluate_single_model(model_info, device, log_file):
    name = model_info['name']
    cfg_path = model_info['cfg']
    weight_path = model_info['weight']

    write_log(f"\n{'='*80}", log_file)
    write_log(f"🚀 开始评估模型: {name}", log_file)
    write_log(f"📄 配置文件: {cfg_path}", log_file)
    write_log(f"📥 权重文件: {weight_path}", log_file)

    if not os.path.exists(weight_path):
        write_log(f"❌ 错误: 找不到权重文件，跳过该模型！", log_file)
        return

    # 1. 获取配置
    args = DummyArgs(cfg_path)
    config = get_config(args)
    config.defrost()
    config.EVAL_MODE = True
    config.freeze()

    # 2. 构建测试集 DataLoader
    transforms = build_transform(is_train=False, config=config)
    try:
        base_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transforms)
    except Exception as e:
        write_log(f"❌ 错误: 读取测试集失败 -> {e}", log_file)
        return

    class_names = base_dataset.classes
    num_classes = len(class_names)

    dataset_val = base_dataset
    if config.MODEL.FUSION.ENABLED:
        write_log("🧬 检测到融合模块开启，挂载 ClinicalDatasetWrapper...", log_file)
        dataset_val = ClinicalDatasetWrapper(
            base_dataset,
            csv_path=CLINICAL_CSV,
            clinical_dim=config.MODEL.FUSION.CLINICAL_DIM
        )

    data_loader_val = DataLoader(
        dataset_val, batch_size=config.DATA.BATCH_SIZE, shuffle=False,
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY
    )

    # 3. 构建模型并加载权重
    config.defrost()
    config.MODEL.NUM_CLASSES = num_classes
    config.freeze()

    model = build_model(config)
    model.to(device)

    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    # 4. 运行推理 (直接复用 engine.py 的 validate)
    write_log("🔥 正在进行网络前向推理...", log_file)
    final_stats = validate(config, data_loader_val, model, class_names=class_names)

    y_true = np.array(final_stats['targets'])
    y_pred = np.array(final_stats['preds'])
    y_score = np.array(final_stats['probs'])

    # 5. 计算各项指标
    # --- 基础指标 ---
    acc = accuracy_score(y_true, y_pred)
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    # --- 特异度 (Specificity) ---
    cm = confusion_matrix(y_true, y_pred)
    spec_per = [calculate_specificity(cm, i) for i in range(num_classes)]
    spec_mac = np.mean(spec_per)

    # --- 标签二值化 (为了算 AUC 和 PR-AUC) ---
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    # 处理2分类的边界情况(以防万一)
    if num_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    # --- AUC 和 PR-AUC (Average Precision) ---
    auc_per = []
    prauc_per = []
    for i in range(num_classes):
        try:
            auc_val = roc_auc_score(y_true_bin[:, i], y_score[:, i])
            prauc_val = average_precision_score(y_true_bin[:, i], y_score[:, i])
        except ValueError:
            auc_val = 0.0
            prauc_val = 0.0
        auc_per.append(auc_val)
        prauc_per.append(prauc_val)

    # 宏平均 AUC 和 PR-AUC
    try:
        auc_mac = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        prauc_mac = average_precision_score(y_true_bin, y_score, average='macro')
    except Exception:
        auc_mac = np.mean(auc_per)
        prauc_mac = np.mean(prauc_per)

    # 6. 格式化输出日志
    write_log(f"\n📊 --- {name} 测试结果报告 ---", log_file)
    write_log(f"{'Class':<12} | {'Precision':<9} | {'Specific.':<9} | {'Recall':<9} | {'F1-Score':<9} | {'AUC':<9} | {'PR-AUC':<9}", log_file)
    write_log("-" * 85, log_file)

    for i, cls_name in enumerate(class_names):
        write_log(f"{cls_name:<12} | {p_per[i]:<9.4f} | {spec_per[i]:<9.4f} | {r_per[i]:<9.4f} | {f1_per[i]:<9.4f} | {auc_per[i]:<9.4f} | {prauc_per[i]:<9.4f}", log_file)

    write_log("-" * 85, log_file)
    write_log(f"{'MACRO-AVG':<12} | {p_mac:<9.4f} | {spec_mac:<9.4f} | {r_mac:<9.4f} | {f1_mac:<9.4f} | {auc_mac:<9.4f} | {prauc_mac:<9.4f}", log_file)
    write_log(f"\n🏆 Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)", log_file)

def main():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    # 打开日志文件准备写入
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as log_file:
        write_log(f"=== 实验综合评估日志 ===", log_file)
        write_log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        write_log(f"测试集目录: {TEST_DATA_PATH}", log_file)

        # 逐个跑模型
        for model_info in MODELS_TO_TEST:
            evaluate_single_model(model_info, device, log_file)

    print(f"\n🎉 所有模型评估完成！完整日志已保存至: {LOG_FILE_PATH}")

if __name__ == '__main__':
    main()