# Swin-Transformer/engine.py

import torch
import torch.nn as nn
from timm.utils import accuracy, AverageMeter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from utils import reduce_tensor, MetricLogger, SmoothedValue

def calculate_per_class_metrics(y_true, y_pred):
    """辅助函数：计算简单的每类指标"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return precision, recall, f1

def calculate_specificity(y_true, y_pred, labels):
    """辅助函数：计算每类的特异度 (Specificity)"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = []
    for i in range(len(labels)):
        # TN = 总样本 - (FP + FN + TP)
        # 简化计算: TN = Total - (Row_i_Sum + Col_i_Sum - TP)
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]

        if tn + fp == 0:
            spec = 0.0
        else:
            spec = tn / (tn + fp)
        specificity.append(spec)
    return specificity

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, class_names=None):
    model.train()
    optimizer.zero_grad()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    all_preds = []
    all_targets = []

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # --- 1. 数据解包与 GPU 搬运 ---
        if config.MODEL.FUSION.ENABLED:
            # samples 为 (images, clinical_data) 元组
            images, clinical_data = samples
            clinical_data = clinical_data.cuda(non_blocking=True)
        else:
            images = samples
            clinical_data = None

        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # 保存原始标签（索引格式），用于 Acc 统计
        # 因为 Mixup 会把 targets 变成概率分布
        targets_orig = targets.clone()

        # --- 2. Mixup 增强 (只针对图像和标签) ---
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        # --- 3. 前向传播 ---
        # 我们的 SwinV2 模型 forward 接口已支持 clinical_data=None 传参
        output = model(images, clinical_data=clinical_data)

        # Loss 计算：如果用了 Mixup，targets 是 Soft Label
        loss = criterion(output, targets)

        # --- 4. 反向传播与优化 ---
        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        optimizer.step()
        lr_scheduler.step_update(epoch * len(data_loader) + idx)

        # --- 5. 指标统计 ---
        # 使用 targets_orig (原始索引) 来计算 Acc，排除 Mixup 干扰
        acc1 = (output.argmax(dim=1) == targets_orig).float().mean()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc=acc1.item())

        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_targets.extend(targets_orig.cpu().numpy())

    # --- 6. Epoch 结束时输出
    print(f"\n--- Train Epoch {epoch} Detailed Metrics ---")
    p, r, f, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)

    # 确保 class_names 和 labels 对应
    unique_labels = range(len(p))
    specs = calculate_specificity(all_targets, all_preds, unique_labels)

    if class_names is None:
        class_names = [str(i) for i in unique_labels]

    # [修改] 增加 Specificity 列
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1-Score':<12}")
    print("-" * 70) # 加长横线
    for i, name in enumerate(class_names):
        if i < len(p):
            # [修改] 增加 specs[i] 输出
            print(f"{name:<15} {p[i]*100:<12.2f} {r[i]*100:<12.2f} {specs[i]*100:<12.2f} {f[i]*100:<12.2f}")

    print("-" * 70)
    acc_val = accuracy_score(all_targets, all_preds)
    print(f"Global Acc: {acc_val * 100:.2f}%\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(config, data_loader, model, class_names=None):
    # 验证时不使用训练时的权重惩罚，使用标准交叉熵
    criterion = nn.CrossEntropyLoss()
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val:'

    all_preds = []
    all_targets = []
    all_probs = []

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # --- 数据解包 ---
        if config.MODEL.FUSION.ENABLED:
            images, clinical_data = samples
            clinical_data = clinical_data.cuda(non_blocking=True)
        else:
            images = samples
            clinical_data = None

        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # --- 前向传播 ---
        output = model(images, clinical_data=clinical_data)

        loss = criterion(output, targets)
        acc1 = (output.argmax(dim=1) == targets).float().mean()

        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc1.item())

        # 收集概率数据用于 ROC/AUC 计算
        probs = torch.softmax(output, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    # --- 输出验证集报告 ---
    # --- 输出验证集报告 ---
    print(f"\n--- Val Detailed Metrics ---")
    p, r, f, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)

    unique_labels = range(len(p))
    specs = calculate_specificity(all_targets, all_preds, unique_labels)

    if class_names is None:
        class_names = [str(i) for i in unique_labels]

    # [修改] 增加 Specificity 列
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1-Score':<12}")
    print("-" * 70)
    for i, name in enumerate(class_names):
        if i < len(p):
            print(f"{name:<15} {p[i]*100:<12.2f} {r[i]*100:<12.2f} {specs[i]*100:<12.2f} {f[i]*100:<12.2f}")
    print("-" * 70 + "\n")

    return {
        'loss': metric_logger.loss.global_avg,
        'acc': metric_logger.acc.global_avg,
        'targets': np.array(all_targets),
        'preds': np.array(all_preds),
        'probs': np.array(all_probs)
    }