# Swin-Transformer/utils.py

import os
import torch
import torch.distributed as dist
import math
import numpy as np
import pandas as pd
import seaborn as sns
import time
import datetime
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score


def find_unique_output_dir(base_output_dir, model_name):
    """
    REQ 5: 自动管理输出目录
    ./output/model_name -> ./output/model_name_1 -> ...
    """
    # 基础路径: ./output/swinv2_fusion
    target_dir = os.path.join(base_output_dir, model_name)

    if not os.path.exists(target_dir):
        return target_dir

    # 如果存在，寻找可用的后缀
    counter = 1
    while True:
        new_dir = f"{target_dir}_{counter}"
        if not os.path.exists(new_dir):
            return new_dir
        counter += 1

def load_checkpoint(config, model, optimizer, lr_scheduler, logger=None):
    """
    智能加载检查点：
    1. 优先加载 config.MODEL.RESUME 指定的路径
    2. 其次尝试 auto_resume (加载最新的 checkpoint)
    3. 最后尝试 config.MODEL.PRETRAINED (加载官方预训练权重)
    """
    if logger is None:
        # 创建一个简单的 dummy logger，把 info/warning 映射到 print
        class PrintLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARN] {msg}")
        logger = PrintLogger()

    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")

    # 1. 尝试 Resume (断点续训)
    if config.MODEL.RESUME:
        path = config.MODEL.RESUME
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location='cpu')

            # 加载模型权重
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            logger.info(msg)

            # 恢复优化器、LR调度器、Epoch
            if 'optimizer' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                config.defrost()
                config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
                config.freeze()

            logger.info(f"=> loaded successfully '{path}' (epoch {checkpoint['epoch']})")
            if 'max_accuracy' in checkpoint:
                return checkpoint['max_accuracy']
        else:
            logger.warning(f"=> no checkpoint found at '{path}'")

    # 2. 尝试加载 Pretrained (迁移学习)
    elif config.MODEL.PRETRAINED:
        path = config.MODEL.PRETRAINED
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

            # --- [核心修改] 智能过滤不匹配的权重 (针对 H-CQA 魔改) ---
            model_dict = model.state_dict()
            # 过滤掉形状不匹配的键 (例如 head.weight, head.bias)
            # 因为我们的 head 已经被替换为 ClinicalHead，维度变了
            filtered_dict = {k: v for k, v in state_dict.items()
                             if k in model_dict and v.shape == model_dict[k].shape}

            removed_keys = [k for k in state_dict.keys() if k not in filtered_dict]
            logger.info(f"⚠️  Pretrained weights mismatch/removed keys: {removed_keys}")

            msg = model.load_state_dict(filtered_dict, strict=False)
            logger.info(f"=> loaded pretrained weights from '{path}'")
            logger.info(f"   Missing keys (expected for new head): {msg.missing_keys}")
            # -----------------------------------------------------
        else:
            logger.warning(f"=> no pretrained file found at '{path}'")

    return 0.0


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger=None):
    if logger is None:
        class PrintLogger:
            def info(self, msg): print(f"[INFO] {msg}")
        logger = PrintLogger()
    """
    保存模型检查点
    """
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'max_accuracy': max_accuracy,
        'epoch': epoch,
        'config': config,
    }

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

    # 另外保存一份 best_checkpoint
    # (逻辑通常在 main 里控制，这里只负责保存指定文件)

class EarlyStopping:
    """
    REQ 4: 早停机制
    """
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_acc, val_loss):
        score = val_acc # 这里的策略是根据 Acc 还是 Loss？通常用 Loss 更稳，但 Acc 更直观。
        # 这里我们结合使用：关注 Acc 提升，如果不提升看 Loss 是否恶化

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'   [EarlyStopping] Counter: {self.counter} / {self.patience} (Best Acc: {self.best_score:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def get_grad_norm(parameters, norm_type=2):
    """
    计算梯度范数 (用于梯度裁剪监控)
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    """
    自动查找 output 目录下最新的 ckpt 用于续训
    """
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        return latest_checkpoint
    return None


def reduce_tensor(tensor):
    """
    分布式训练工具：汇总所有 GPU 上的 tensor 值并取平均
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

class SmoothedValue(object):
    """
    跟踪一系列数值，并提供对它们的平滑处理（平均值、全局平均值等）。
    用于在训练日志中显示 Loss 平滑下降的曲线。
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        分布式训练同步工具
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """
    日志记录器：负责管理多个 SmoothedValue，并打印进度条。
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'MetricLogger' object has no attribute '{}'".format(attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def accuracy(output, target, topk=(1,)):
    """
    计算 Top-k 准确率
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

# =========================================================
# 3. 绘图与分析工具 (REQ 6)
# =========================================================

def plot_loss_curve(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_scores, classes, output_dir):
    # y_true: (N,), y_scores: (N, NumClasses) (softmax probabilities)
    # 需要处理 One-hot
    from sklearn.preprocessing import label_binarize
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (class {classes[i]}) (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def generate_classification_report(y_true, y_pred, classes, output_dir):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    print("\n====== Final Classification Report ======")
    print(df)
    return df