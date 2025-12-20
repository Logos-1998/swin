# Swin-Transformer/engine.py

import torch
import torch.nn as nn
from timm.utils import accuracy, AverageMeter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import reduce_tensor, MetricLogger, SmoothedValue

def calculate_per_class_metrics(y_true, y_pred):
    """辅助函数：计算简单的每类指标"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return precision, recall, f1

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, class_names=None):
    model.train()
    optimizer.zero_grad()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    all_preds = []
    all_targets = []

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # 1. 数据解包
        if config.MODEL.FUSION.ENABLED:
            images, clinical_data = samples
            images = images.cuda(non_blocking=True)
            clinical_data = clinical_data.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        else:
            images = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            clinical_data = None
            if mixup_fn is not None:
                images, targets_mix = mixup_fn(images, targets)

        # 2. 前向传播
        if config.MODEL.FUSION.ENABLED:
            output = model(images, clinical_data)
        else:
            output = model(images)

        # 3. Loss 计算 (注意 Mixup 对 Label 的影响，这里简化处理，只监控 Raw Target)
        # 如果用了 Mixup，targets_mix 用于算 Loss，但监控指标用原始 targets
        loss = criterion(output, targets if mixup_fn is None else targets_mix)

        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        optimizer.step()
        lr_scheduler.step_update(epoch * len(data_loader) + idx)

        # 4. 收集数据用于 Epoch 结束时的统计
        acc1 = (output.argmax(dim=1) == targets).float().mean()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc=acc1.item())

        # 收集预测结果 (CPU)
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    # REQ 7: Epoch 结束时输出训练集各类别指标
    print(f"\n--- Train Epoch {epoch} Detailed Metrics ---")
    p, r, f, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)

    # 如果没传 class_names，就用数字兜底
    if class_names is None:
        class_names = [str(i) for i in range(len(p))]

    # 为了排版美观，使用字典格式打印，或者简单的对齐打印
    # 这里使用简单的对齐打印
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 55)
    for i, name in enumerate(class_names):
        # 防止类别数不匹配的边界情况
        if i < len(p):
            print(f"{name:<15} {p[i]*100:<12.2f} {r[i]*100:<12.2f} {f[i]*100:<12.2f}")

    print("-" * 45)
    acc_val = accuracy_score(all_targets, all_preds)
    # [修正] Global Acc 显示百分比
    print(f"Global Acc: {acc_val * 100:.2f}%\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(config, data_loader, model, class_names=None):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val:'

    all_preds = []
    all_targets = []
    all_probs = [] # 存储概率用于 ROC

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if config.MODEL.FUSION.ENABLED:
            images, clinical_data = samples
            images = images.cuda(non_blocking=True)
            clinical_data = clinical_data.cuda(non_blocking=True)
        else:
            images = samples.cuda(non_blocking=True)
            clinical_data = None

        targets = targets.cuda(non_blocking=True)

        if config.MODEL.FUSION.ENABLED:
            output = model(images, clinical_data)
        else:
            output = model(images)

        loss = criterion(output, targets)
        acc1 = (output.argmax(dim=1) == targets).float().mean()

        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc1.item())

        # 收集数据
        probs = torch.softmax(output, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    # --- [修复] 优化输出显示 ---
    print(f"\n--- Val Detailed Metrics ---")
    p, r, f, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)

    if class_names is None:
        class_names = [str(i) for i in range(len(p))]

    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 55)
    for i, name in enumerate(class_names):
        if i < len(p):
            print(f"{name:<15} {p[i]*100:<12.2f} {r[i]*100:<12.2f} {f[i]*100:<12.2f}")
    print("-" * 55 + "\n")

    return {
        'loss': metric_logger.loss.global_avg,
        'acc': metric_logger.acc.global_avg,
        'targets': np.array(all_targets),
        'preds': np.array(all_preds),
        'probs': np.array(all_probs)
    }