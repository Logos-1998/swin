import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision import datasets
from torch.utils.data import DataLoader

# ================= 引入项目组件 =================
from config import get_config
from models import build_model
from data.build import build_transform
from engine import validate

try:
    from data.build import ClinicalDatasetWrapper
except ImportError:
    pass

# ================= 1. 自定义设置区 =================

# --- 数据集与输出 ---
TEST_DIR = r"D:\Documents\Swin-Transformer\dataset\test"
CLINICAL_CSV = r"D:\Documents\Swin-Transformer\dataset\clinical_data.csv"
OUTPUT_DIR = r"D:\Documents\Swin-Transformer\results"

# --- 核心：你要重点画哪个类别的 PR 曲线？---
# (必须是 dataset/test/ 下的真实子文件夹名，通常为 'OPA' 或 'OP')
TARGET_CLASS = "Normal"

# --- 真实模型配置区 ---
# 请填入你想要对比的模型的名称、配置文件(.yaml)和权重文件(.pth)
# 颜色、线型和粗细可随意自定义
MODELS_TO_TEST = [
    {
        "label": "PR曲线",
        "cfg": r"D:\Documents\Swin-Transformer\configs\exp3_fusion.yaml",
        "weight": r"D:\Documents\Swin-Transformer\output\exp4_full\checkpoint_middle.pth",
        "color": "blue", "style": "--", "width": 3,#Normal
        #"color": "red", "style": "--", "width": 3,#OP
        #"color": "green", "style": "--", "width": 3,#OPA
    },

]

# --- 绘图参数配置 ---
X_AXIS_LABEL = '召回率'
Y_AXIS_LABEL = '精确率'
AXIS_LABEL_FONT_SIZE = 30
TICK_LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 27
# ---> 新增：控制随机分类器横向虚线的粗细 <---
BASELINE_LINE_WIDTH = 2.5

# ================= 2. 辅助类与字体配置 =================

# 构造假的 args 对象，用于加载项目 config
class DummyArgs:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = None
        self.batch_size = 32
        self.data_path = TEST_DIR
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

# 全局字体设为 Times New Roman (自动接管全局英文和数字)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 坐标轴的宋体对象
chinese_font = FontProperties(family='SimSun', size=AXIS_LABEL_FONT_SIZE)
# 专门给图例配置的混合字体（优先使用新罗马，遇到中文自动回退到宋体）
legend_font = FontProperties(family=['Times New Roman', 'SimSun'], size=LEGEND_FONT_SIZE)

# ================= 3. 核心推理函数 =================

def run_inference_for_model(model_info, device):
    """ 加载单个模型的配置和权重，在测试集上运行推理，返回预测概率和真实标签 """
    print(f"\n🚀 开始评估模型: {model_info['label']}")

    # 1. 解析配置
    args = DummyArgs(model_info['cfg'])
    config = get_config(args)
    config.defrost()
    config.EVAL_MODE = True
    config.freeze()

    # 2. 构建测试集数据管道
    transforms = build_transform(is_train=False, config=config)
    base_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transforms)
    class_names = base_dataset.classes
    num_classes = len(class_names)

    # 如果有开启临床融合，包裹上 DatasetWrapper
    dataset_val = base_dataset
    if config.MODEL.FUSION.ENABLED:
        print("   🧬 检测到融合模块，正在挂载临床数据...")
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

    try:
        checkpoint = torch.load(model_info['weight'], map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("   ✅ 权重加载成功！")
    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")
        return None, None, None

    # 4. 执行推理获取真实矩阵
    print("   🔥 正在运行全测试集前向推理...")
    final_stats = validate(config, data_loader_val, model, class_names=class_names)

    # 转换为 numpy
    targets = np.array(final_stats['targets'])
    probs = np.array(final_stats['probs'])

    return class_names, targets, probs

# ================= 4. 主程序 =================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  运行设备: {device}")

    # 初始化画布
    plt.figure(figsize=(9, 8))

    target_class_idx = None
    baseline_y = 0.0

    print("=" * 65)
    print(f"{'Curve Name':<20} | {'True AP (AUPRC)'}")
    print("-" * 65)

    # 遍历并评估所有模型
    for idx, model_info in enumerate(MODELS_TO_TEST):
        class_names, targets, probs = run_inference_for_model(model_info, device)

        if class_names is None:
            continue

        # 如果是第一个模型，根据其加载的数据集确定目标类别的索引和 Baseline
        if idx == 0:
            if TARGET_CLASS not in class_names:
                raise ValueError(f"测试集中未找到目标类别: {TARGET_CLASS}，现有类别: {class_names}")
            target_class_idx = class_names.index(TARGET_CLASS)

            # 计算 Baseline: 目标类样本数 / 总样本数
            n_pos = np.sum(targets == target_class_idx)
            n_total = len(targets)
            baseline_y = n_pos / n_total
            print(f"\n📊 类别解析完成: {TARGET_CLASS} (索引={target_class_idx})")
            print(f"   测试集总计 {n_total} 张, 目标类 {n_pos} 张 -> Baseline = {baseline_y:.4f}\n")

        # 将目标类别二值化 (一对多)
        y_true_bin = (targets == target_class_idx).astype(int)
        # 获取模型对目标类别的预测概率
        y_score = probs[:, target_class_idx]

        # 真实计算 AP 和 PR 坐标点
        ap = average_precision_score(y_true_bin, y_score)
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score)

        print(f"{model_info['label']:<20} | {ap:.4f}")

        # 将真实曲线画到图上
        label_text = f"{model_info['label']} (AP = {ap:.3f})"
        plt.plot(recall, precision,
                 label=label_text,
                 color=model_info['color'],
                 linestyle=model_info['style'],
                 linewidth=model_info['width'],
                 alpha=0.9)

    # ================= 5. 装饰与保存 =================
    # ---> 修改项：改为“随机分类器”，并应用设定的线宽 <---
    plt.axhline(y=baseline_y, color='gray', linestyle='--', lw=BASELINE_LINE_WIDTH, label=f'随机分类器 ({baseline_y:.3f})')

    # 设置轴范围
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])

    # 应用坐标轴标签 (独立的宋体)
    plt.xlabel(X_AXIS_LABEL, fontproperties=chinese_font)
    plt.ylabel(Y_AXIS_LABEL, fontproperties=chinese_font)

    # 自定义刻度数字大小 (全局 Times New Roman)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONT_SIZE)

    # 图例挂载专用混合字体对象 (使用 prop 参数)
    plt.legend(loc="lower left", frameon=True, prop=legend_font)

    # 密网格设计
    plt.minorticks_on()
    plt.grid(visible=True, which='major', linestyle='--', color='gray', alpha=0.5)
    plt.grid(visible=True, which='minor', linestyle=':', color='gray', alpha=0.4)

    plt.tight_layout()

    # 保存高清图
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'real_pr_curve_{TARGET_CLASS}.jpg')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')

    print(f"\n🎉 真实 PR 曲线生成完毕！高清大图已保存至：\n👉 {save_path}")
    plt.show()

if __name__ == '__main__':
    main()