# tools/calculate_mean_std.py

import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def get_mean_std(data_dir, batch_size=128, num_workers=4):
    """
    计算数据集的 Mean 和 Std (RGB通道)
    """
    print(f"Checking directory: {data_dir}")

    # 只需要 ToTensor，不需要其他增强，因为我们要算原始像素的统计值
    # ToTensor 会自动把像素值从 [0, 255] 缩放到 [0.0, 1.0]
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # 统一大小，防止某些超大图甚至显存
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    print(f"Starting calculation on {len(dataset)} images...")
    start_time = time.time()

    for i, (images, _) in enumerate(loader):
        # images shape: [Batch, 3, H, W]
        batch_samples = images.size(0) # batch size (might be smaller for last batch)
        images = images.view(batch_samples, images.size(1), -1) # [B, 3, H*W]

        # 累加 Batch 的均值和方差
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

        total_samples += batch_samples

        if i % 10 == 0:
            print(f"Processed batch {i}...")

    # 计算全局平均
    mean /= total_samples
    std /= total_samples

    print(f"\nCalculation finished in {time.time() - start_time:.2f}s")
    print("="*40)
    print(f"Dataset: {data_dir}")
    print(f"MEAN: {mean.tolist()}")
    print(f"STD:  {std.tolist()}")
    print("="*40)

    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认指向 dataset/train
    parser.add_argument('--data-path', type=str, default='dataset/train', help='path to training dataset')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        # 尝试修正路径 (兼容从根目录运行的情况)
        if os.path.exists(os.path.join('../dataset', 'train')):
            args.data_path = '../dataset/train'
        elif os.path.exists(os.path.join('dataset', 'train')):
            args.data_path = 'dataset/train'
        else:
            print(f"Error: Path {args.data_path} not found.")
            exit(1)

    get_mean_std(args.data_path)