# Swin-Transformer/data/build.py

import os
import torch
import numpy as np
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