# Swin-Transformer/data/build.py

import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

class ClinicalDatasetWrapper(torch.utils.data.Dataset):
    """
    包装器：同时返回 (Image, ClinicalData) 和 Target
    """
    def __init__(self, dataset, csv_path, clinical_dim=4):
        self.dataset = dataset
        self.csv_path = csv_path

        print(f"====== [Dataset] Initializing Clinical Data Fusion ======")
        print(f"Loading CSV from: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Critical Error: Clinical CSV not found at {csv_path}")

        # 读取 CSV
        df = pd.read_csv(csv_path)

        # -----------------------------------------------------------
        # [核心修复] 明确指定要使用的列名，不再使用索引切片
        # 你的 CSV 结构: image_path, patient_id, age, gender, height, weight
        # -----------------------------------------------------------
        # 这种写法最稳健，不管你前面插入多少个ID列，都不会错
        target_features = ['age', 'gender', 'height', 'weight']

        # 检查这些列是否都在 CSV 里
        missing_cols = [c for c in target_features if c not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV is missing required columns: {missing_cols}")

        feature_cols = target_features
        print(f"Target clinical features ({len(feature_cols)}): {list(feature_cols)}")

        # 简单的数据清洗与归一化 (Z-Score)
        for col in feature_cols:
            # 确保是数值类型
            if pd.api.types.is_numeric_dtype(df[col]):
                # 填充 NaN
                df[col] = df[col].fillna(df[col].mean())
                # 归一化
                mean, std = df[col].mean(), df[col].std()
                if std > 1e-6:
                    df[col] = (df[col] - mean) / std
            else:
                # 非数值类型 (如 'M'/'F') 简单编码
                df[col] = df[col].astype('category').cat.codes

        # -----------------------------------------------------------
        # 建立索引映射: Filename (Base) -> Clinical Tensor
        # -----------------------------------------------------------
        self.clin_map = {}
        for idx, row in df.iterrows():
            # 获取纯文件名 (去除路径) 用于匹配
            # 你的第0列是 image_path
            fname = os.path.basename(str(row['image_path']))

            # 提取特征向量
            feats = row[feature_cols].values.astype(np.float32)
            self.clin_map[fname] = torch.tensor(feats, dtype=torch.float32)

        print(f"Successfully mapped {len(self.clin_map)} clinical records.")

    def __getitem__(self, index):
        # 1. 获取原始图片和标签
        img, target = self.dataset[index]

        # 2. 获取文件名
        # 不同 Dataset 获取文件名的方式不同，这里做兼容处理
        filename = "unknown"
        if hasattr(self.dataset, 'samples'):
            # ImageFolder 或 CachedImageFolder
            full_path = self.dataset.samples[index][0]
            filename = os.path.basename(full_path)
        elif hasattr(self.dataset, 'imgs'):
            full_path = self.dataset.imgs[index][0]
            filename = os.path.basename(full_path)

        # 3. 查找临床特征
        if filename in self.clin_map:
            clin_data = self.clin_map[filename]
        else:
            # 如果找不到，返回全0向量 (代表平均人)
            # print(f"Warning: No clinical data for {filename}")
            clin_data = torch.zeros(len(next(iter(self.clin_map.values()))), dtype=torch.float32)

        # 返回格式：((img, clin_data), target)
        # 注意：这样修改后，Mixup 和 Training Loop 需要相应调整！
        return (img, clin_data), target

    def __len__(self):
        return len(self.dataset)

def build_loader(config):
    # 1. 构建数据集
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()

    print(f"✅ [单卡模式] 训练集构建完成，样本数: {len(dataset_train)}，类别数: {config.MODEL.NUM_CLASSES}")

    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"✅ [单卡模式] 验证集构建完成，样本数: {len(dataset_val)}")

    # 2. 构建采样器 (Sampler) - 纯单卡逻辑
    # 训练集：随机打乱
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # 验证集：根据配置决定是否打乱，安全获取参数
    use_shuffle = getattr(config.TEST, 'SHUFFLE', False)
    if use_shuffle:
        sampler_val = torch.utils.data.RandomSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 3. 构建 DataLoader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # 4. 设置 Mixup 数据增强 (修复 CUTMIX_MINMAX 报错)
    mixup_fn = None
    # 安全获取 CUTMIX_MINMAX，如果没有则为 None
    cutmix_minmax = getattr(config.AUG, 'CUTMIX_MINMAX', None)

    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or cutmix_minmax is not None

    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=cutmix_minmax, # 使用安全获取的变量
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

    # 优先使用 CachedImageFolder (如果配置了 ZIP)，否则用标准 ImageFolder
    if config.DATA.ZIP_MODE:
        ann_file = prefix + "_map.txt"
        prefix = prefix + ".zip@/"
        dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                    cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
    else:
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)

    # 自动获取类别数
    if hasattr(dataset, 'classes'):
        nb_classes = len(dataset.classes)
    else:
        nb_classes = 1000

    # [新增逻辑] 临床数据融合包装 (H-CQA Support)
    # =================================================================
    if config.MODEL.FUSION.ENABLED:
        csv_path = config.MODEL.FUSION.CSV_PATH
        # 如果配置里没写路径，使用默认值
        if not csv_path:
            csv_path = os.path.join('toolkit', 'clinical_data.csv')

        # 只有当 CSV 文件真的存在时才包装，否则报错或警告
        if os.path.exists(csv_path):
            print(f"====== [Build Dataset] Wrapping dataset with Clinical Data ({prefix}) ======")
            dataset = ClinicalDatasetWrapper(
                dataset,
                csv_path=csv_path,
                clinical_dim=config.MODEL.FUSION.CLINICAL_DIM
            )
        else:
            raise FileNotFoundError(f"Fusion enabled but CSV not found at: {csv_path}")
    # =================================================================

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        # 安全获取 TEST.CROP 参数
        use_crop = getattr(config.TEST, 'CROP', True)
        if use_crop:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)