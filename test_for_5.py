import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import confusion_matrix

# 复用项目模块
from config import get_config
from models import build_model
from data.build import build_transform  # 只复用 transform 构建逻辑

# ==============================================================================
# 🛠️ 用户配置区
# ==============================================================================

# 1. 权重路径
WEIGHT_PATH = r"D:\Documents\Swin-Transformer\output\exp4_full\checkpoint_best.pth"

# 2. 配置文件路径
CONFIG_PATH = r"D:\Documents\Swin-Transformer\configs\exp3_fusion.yaml"

# 3. 测试集路径 (文件夹下直接是类别子文件夹，例如 dataset/test/0_normal)
TEST_DATA_PATH = r"D:\Documents\Data\K_for_5\1\test"

# 4. 结果保存目录
OUTPUT_ROOT = r"D:\Documents\Swin-Transformer\results\base"
SUB_FOLDER_NAME = "baseline"

# ==============================================================================

class VertebraImageFolder(datasets.ImageFolder):
    """
    重写 ImageFolder，除了返回图片张量和标签，还通过解析文件名返回椎体的唯一标识符
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # 提取 A10022380_L1_1.7_338.png 中的 A10022380_L1 作为唯一标识符
        filename = os.path.basename(path)
        parts = filename.split('_')
        if len(parts) >= 2:
            vertebra_id = f"{parts[0]}_{parts[1]}"
        else:
            vertebra_id = filename

        return sample, target, vertebra_id

def calculate_vertebra_metrics(y_true, y_pred, classes):
    """
    基于聚合后的椎体分类结果计算各类别评估参数
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    print("\n" + "🚀"*20)
    print("各类别细粒度评估报告 (基于椎体级别聚合):")
    print("🚀"*20)

    for i, class_name in enumerate(classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        accuracy = (TP + TN) / cm.sum() if cm.sum() > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"类别: {class_name}")
        print(f"  - 样本数 (椎体总数): {cm[i, :].sum()}")
        print(f"  - 准确率 (Accuracy) : {accuracy:.4f}")
        print(f"  - 精准率 (Precision): {precision:.4f}")
        print(f"  - 召回率 (Recall)   : {recall:.4f}")
        print(f"  - F1-Score        : {f1_score:.4f}")
        print("-" * 50)


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

    # --- 3. 手动构建测试集加载器 (采用自定义的 VertebraImageFolder) ---
    print(f"⏳ Loading test data directly from: {TEST_DATA_PATH}")

    transforms = build_transform(is_train=False, config=config)

    try:
        dataset_val = VertebraImageFolder(root=TEST_DATA_PATH, transform=transforms)
    except FileNotFoundError:
        print(f"❌ Error: 路径 {TEST_DATA_PATH} 不存在或不包含图片类别文件夹。")
        return

    class_names = dataset_val.classes
    print(f"🏷️ Classes found ({len(class_names)}): {class_names}")

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

    # --- 5. 加载权重 (加入 weights_only=False 规避 PyTorch 2.6 安全警告) ---
    print(f"📥 Loading checkpoint: {WEIGHT_PATH}")
    if not os.path.isfile(WEIGHT_PATH):
        print("❌ Checkpoint file not found!")
        return

    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # 推荐此处后续改为 strict=True 严格约束
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # --- 6. 核心推理：推断切片概率并记录 ---
    print("🔥 Starting inference and collecting slice probabilities...")
    model.eval()

    vertebra_preds = defaultdict(list)
    vertebra_labels = {}

    with torch.no_grad():
        # 注意这里获取到的是三个返回值：图片张量、标签、椎体标识符
        for images, targets, vertebra_ids in data_loader_val:
            images = images.to(device, non_blocking=True)
            outputs = model(images)

            # 使用 Softmax 生成切片级的概率分布
            probs = F.softmax(outputs, dim=-1).cpu().numpy()
            targets_np = targets.numpy()

            # 存入对应椎体 ID 的字典中
            for i in range(len(vertebra_ids)):
                v_id = vertebra_ids[i]
                vertebra_preds[v_id].append(probs[i])
                # 同一个椎体的所有切片处于同一目录，标签相同，故直接更新即可
                vertebra_labels[v_id] = targets_np[i]

    # --- 7. 数据聚合与评估 ---
    print("📊 Executing Average Pooling for Vertebra-level probabilities...")
    y_true_vertebra = []
    y_pred_vertebra = []

    for v_id, prob_list in vertebra_preds.items():
        # 将单个椎体的所有切片概率堆叠为矩阵 (N, num_classes)
        prob_matrix = np.array(prob_list)
        # 沿列方向 (axis=0) 做平均池化
        mean_prob = np.mean(prob_matrix, axis=0)
        # 求出最终预测结果
        final_pred_class = np.argmax(mean_prob)

        y_true_vertebra.append(vertebra_labels[v_id])
        y_pred_vertebra.append(final_pred_class)

    # 计算并打印评估指标
    calculate_vertebra_metrics(y_true_vertebra, y_pred_vertebra, class_names)
    print("\n✅ Done!")

if __name__ == '__main__':
    main()