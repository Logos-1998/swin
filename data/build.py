# Swin-Transformer/data/build.py

import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from torchvision.transforms import InterpolationMode # [修改] 确保引入

from .cached_image_folder import CachedImageFolder
from torch.utils.data import Dataset
import os

class VertebraDatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

        # 新增以下三行：将基础数据集的核心属性显式暴露出来
        self.classes = getattr(base_dataset, 'classes', None)
        self.targets = getattr(base_dataset, 'targets', None)
        self.samples = getattr(base_dataset, 'samples', None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample, target = self.base_dataset[index]
        # 原版 ImageFolder 的路径保存在 samples 属性中
        path = self.base_dataset.samples[index][0]
        filename = os.path.basename(path)
        parts = filename.split('_')
        vertebra_id = f"{parts[0]}_{parts[1]}"

        return sample, target, vertebra_id

def _pil_interp(method):
    """
    辅助函数：将配置字符串映射为 InterpolationMode 枚举。
    注意：Lanczos 不支持旋转等几何变换。
    """
    method = method.lower()
    if method == 'bicubic':
        return InterpolationMode.BICUBIC
    elif method == 'lanczos':
        # 在这里强制拦截：为了防止增强（如旋转）报错，统一返回 BICUBIC
        # 真正的 LANCZOS 我们只会在验证集的纯 Resize 中手动开启
        return InterpolationMode.BICUBIC
    elif method == 'hamming':
        return InterpolationMode.HAMMING
    else:
        return InterpolationMode.BILINEAR

# 猴子补丁：覆盖 timm 内部默认插值选择器
import timm.data.transforms as timm_transforms
timm_transforms._pil_interp = _pil_interp

class ClinicalDatasetWrapper(torch.utils.data.Dataset):
    """
    包装器：同时返回 (Image, ClinicalData) 和 Target
    """
    def __init__(self, dataset, csv_path, clinical_dim=4):
        self.dataset = dataset
        self.csv_path = csv_path
        self.clinical_dim = clinical_dim

        print(f"====== [Dataset] Initializing Clinical Data Fusion ======")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Critical Error: Clinical CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)
        target_features = ['age', 'gender', 'height', 'weight']

        # 简单的数据清洗与归一化
        for col in target_features:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                    mean, std = df[col].mean(), df[col].std()
                    if std > 1e-6:
                        df[col] = (df[col] - mean) / std
                else:
                    df[col] = df[col].astype('category').cat.codes
            else:
                df[col] = 0.0

        self.clin_map = {}
        for idx, row in df.iterrows():
            fname = os.path.basename(str(row['image_path']))
            feats = row[target_features].values.astype(np.float32)
            self.clin_map[fname] = torch.tensor(feats, dtype=torch.float32)

        print(f"Successfully mapped {len(self.clin_map)} clinical records.")

    def __getitem__(self, index):
        # 1. 修正拆包：接收底层 Wrapper 传来的 3 个参数
        img, target, vertebra_id = self.dataset[index]

        filename = "unknown"
        # 2. 修正路径穿透：由于 dataset 外面包了一层 VertebraDatasetWrapper，必须通过 .base_dataset 访问 .samples
        if hasattr(self.dataset, 'base_dataset') and hasattr(self.dataset.base_dataset, 'samples'):
            filename = os.path.basename(self.dataset.base_dataset.samples[index][0])
        elif hasattr(self.dataset, 'samples'):
            filename = os.path.basename(self.dataset.samples[index][0])
        elif hasattr(self.dataset, 'imgs'):
            filename = os.path.basename(self.dataset.imgs[index][0])

        # 获取临床数据
        clin_data = self.clin_map.get(filename, torch.zeros(self.clinical_dim, dtype=torch.float32))

        # 3. 修正返回端：将 vertebra_id 加在末尾
        return (img, clin_data), target, vertebra_id

    def __len__(self):
        return len(self.dataset)

def build_loader(config):
    config.defrost()
    # 先获取数据集
    dataset_train, nb_classes = build_dataset(is_train=True, config=config)
    # 再更新类别数
    config.MODEL.NUM_CLASSES = nb_classes
    config.freeze()

    print(f"✅ 训练集构建完成，样本数: {len(dataset_train)}，类别数: {config.MODEL.NUM_CLASSES}")

    dataset_val, _ = build_dataset(is_train=False, config=config)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    mixup_fn = None
    if config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0.:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    prefix = 'train' if is_train else 'val'

    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
    dataset = VertebraDatasetWrapper(dataset)
    nb_classes = len(dataset.classes)

    if config.MODEL.FUSION.ENABLED:
        csv_path = config.MODEL.FUSION.CSV_PATH or os.path.join('dataset', 'clinical_data.csv')
        dataset = ClinicalDatasetWrapper(
            dataset, csv_path=csv_path, clinical_dim=config.MODEL.FUSION.CLINICAL_DIM
        )

    return dataset, nb_classes

def build_transform(is_train, config):
    # [关键修复] 训练阶段包含几何变换（旋转），绝不能传 'lanczos' 字符串给 timm
    interp_str = config.DATA.INTERPOLATION.lower()
    train_interp = 'bicubic' if interp_str == 'lanczos' else interp_str

    mean = config.DATA.MEAN if config.DATA.MEAN else IMAGENET_DEFAULT_MEAN
    std = config.DATA.STD if config.DATA.STD else IMAGENET_DEFAULT_STD

    if is_train:
        return create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            interpolation=train_interp, # 传入安全值
            mean=mean, std=std,
        )

    # 验证集：不涉及旋转，只有 Resize。此时可以安全使用真正的 LANCZOS。
    val_interp = InterpolationMode.LANCZOS if interp_str == 'lanczos' else _pil_interp(interp_str)

    return transforms.Compose([
        transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), interpolation=val_interp),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])