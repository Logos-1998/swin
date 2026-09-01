import os
import sys
import torch
import argparse
import numpy as np
from collections import defaultdict
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score, confusion_matrix, f1_score

# 引入评估指标库
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize

from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score, confusion_matrix, f1_score

# 将父目录加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config
from models import build_model
from data.build import build_transform, VertebraDatasetWrapper
from utils import plot_confusion_matrix, plot_roc_curve, generate_classification_report

# ==============================================================================
# 🛠️ 第一部分：训练与验证引擎 (供 main.py 调用)
# ==============================================================================

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, class_names=None):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    total_loss = 0.0

    # 引入 tqdm 进度条
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} Train", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        if len(batch) == 3:
            samples, targets, _ = batch
        else:
            samples, targets = batch

        if isinstance(samples, (list, tuple)):
            samples = [s.cuda(non_blocking=True) for s in samples]
        else:
            samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if isinstance(samples, (list, tuple)):
            outputs = model(*samples)
        else:
            outputs = model(samples)

        loss = criterion(outputs, targets)
        loss.backward()

        if config.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step_update(epoch * num_steps + step)
        total_loss += loss.item()

        # 进度条后缀实时显示 Loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return {'loss': total_loss / num_steps}


@torch.no_grad()
def validate(config, data_loader, model, class_names=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    all_logits = []
    all_targets = []
    all_vid = []
    total_loss = 0.0

    # 验证集进度条
    pbar = tqdm(data_loader, desc="Validation", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        if len(batch) == 3:
            images, targets, vertebra_ids = batch
        else:
            raise ValueError("DataLoader 必须返回 3 个元素以支持椎体级验证: images, targets, vertebra_ids")

        if isinstance(images, (list, tuple)):
            images = [img.cuda(non_blocking=True) for img in images]
        else:
            images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if isinstance(images, (list, tuple)):
            output = model(*images)
        else:
            output = model(images)

        loss = criterion(output, targets)
        total_loss += loss.item() * targets.size(0)

        all_logits.append(output.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_vid.extend(vertebra_ids)

    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # ---------------- 椎体级聚合 ----------------
    vid_to_logits = defaultdict(list)
    vid_to_target = {}

    for i, vid in enumerate(all_vid):
        vid_to_logits[vid].append(all_logits[i])
        vid_to_target[vid] = all_targets[i]

    final_targets = []
    final_preds = []
    final_probs = []

    for vid, logits_list in vid_to_logits.items():
        mean_logits = np.mean(logits_list, axis=0)
        probs = torch.softmax(torch.tensor(mean_logits), dim=0).numpy()
        pred = np.argmax(probs)

        final_targets.append(vid_to_target[vid])
        final_preds.append(pred)
        final_probs.append(probs)

    final_targets = np.array(final_targets)
    final_preds = np.array(final_preds)
    final_probs = np.array(final_probs)

    avg_loss = total_loss / len(all_targets)

    # ---------------- 计算细化评估指标 ----------------
    acc = accuracy_score(final_targets, final_preds)

    # 细化计算：获取每一类的指标
    precision_per_class = precision_score(final_targets, final_preds, average=None, zero_division=0)
    recall_per_class = recall_score(final_targets, final_preds, average=None, zero_division=0)
    f1_per_class = f1_score(final_targets, final_preds, average=None, zero_division=0)

    # 宏观计算
    macro_precision = precision_score(final_targets, final_preds, average='macro', zero_division=0)
    macro_recall = recall_score(final_targets, final_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(final_targets, final_preds, average='macro', zero_division=0)

    # 打印排版清晰的类别级性能表
    print("\n" + "="*55)
    print(f"{'Vertebra-Level Classification Report':^55}")
    print("="*55)
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)

    if class_names:
        for i, class_name in enumerate(class_names):
            if i < len(f1_per_class):
                print(f"{class_name:<15} | {precision_per_class[i]:<10.4f} | {recall_per_class[i]:<10.4f} | {f1_per_class[i]:<10.4f}")

    print("-" * 55)
    print(f"{'Macro Avg':<15} | {macro_precision:<10.4f} | {macro_recall:<10.4f} | {macro_f1:<10.4f}")
    print(f"{'Accuracy':<15} | {'':<10} | {'':<10} | {acc:<10.4f}")
    print("="*55)

    return {
        'acc': acc,
        'loss': avg_loss,
        'targets': final_targets,
        'preds': final_preds,
        'probs': final_probs,
        'f1_per_class': f1_per_class  # <--- 将每一类的F1数组传给 main.py
    }

# ==============================================================================
# 🚀 第二部分：独立测试脚本入口
# ==============================================================================
WEIGHT_PATH = r"D:\Documents\Swin-Transformer\K_for_5\output\exp3_fusion\checkpoint_best.pth"
CONFIG_PATH = r"D:\Documents\Swin-Transformer\configs\exp2_esc.yaml"
TEST_DATA_PATH = r"D:\Documents\Data\K\2\test"
OUTPUT_ROOT = r"D:\Documents\Swin-Transformer\K_for_5\results\fusion"
SUB_FOLDER_NAME = "fold_1_eval"

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cfg', type=str, default=CONFIG_PATH)
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-path', type=str, default=TEST_DATA_PATH)
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
    parser.add_argument('--output', default=OUTPUT_ROOT, type=str)
    return parser

def main():
    args, _ = get_args_parser().parse_known_args()
    config = get_config(args)

    config.defrost()
    config.DATA.DATA_PATH = TEST_DATA_PATH
    config.MODEL.RESUME = WEIGHT_PATH
    config.EVAL_MODE = True
    config.OUTPUT = os.path.join(OUTPUT_ROOT, SUB_FOLDER_NAME)
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = build_transform(is_train=False, config=config)
    try:
        dataset_val = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transforms)
        class_names = dataset_val.classes
        dataset_val = VertebraDatasetWrapper(dataset_val)

        if config.MODEL.FUSION.ENABLED:
            from data.build import ClinicalDatasetWrapper
            csv_path = config.MODEL.FUSION.CSV_PATH or os.path.join(TEST_DATA_PATH, 'clinical_data.csv')
            dataset_val = ClinicalDatasetWrapper(
                dataset_val, csv_path=csv_path, clinical_dim=config.MODEL.FUSION.CLINICAL_DIM
            )
    except FileNotFoundError:
        print(f"❌ Error: 路径 {TEST_DATA_PATH} 不存在。")
        return

    data_loader_val = DataLoader(
        dataset_val, batch_size=config.DATA.BATCH_SIZE, shuffle=False,
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY
    )

    config.defrost()
    config.MODEL.NUM_CLASSES = len(class_names)
    config.freeze()

    model = build_model(config)
    model.to(device)

    if not os.path.isfile(WEIGHT_PATH):
        return

    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    final_stats = validate(config, data_loader_val, model, class_names=class_names)

    targets = final_stats['targets']
    preds = final_stats['preds']
    probs = final_stats['probs']

    # 保存 PR 曲线所需数据 (npz格式)
    pr_npz_path = os.path.join(config.OUTPUT, 'pr_data.npz')
    np.savez(pr_npz_path, targets=targets, probs=probs, class_names=class_names)
    print(f"💾 PR 曲线数据已保存至: {pr_npz_path}")

    try:
        plot_confusion_matrix(targets, preds, class_names, config.OUTPUT)
        plot_roc_curve(targets, probs, class_names, config.OUTPUT)
        # 兼容旧版的 Classification Report，依然打印作为对比
        generate_classification_report(targets, preds, class_names, config.OUTPUT)
    except Exception as e:
        print(f"⚠️ Error during plotting: {e}")

if __name__ == '__main__':
    main()