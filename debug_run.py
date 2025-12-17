import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from config import get_config
from models.build import build_model
from data.build import build_loader

# 忽略不重要的警告，保持输出清爽
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def print_model_hierarchy(model):
    """
    可视化打印模型的层级结构
    """
    print("\n" + "="*60)
    print("🏗️  MODEL ARCHITECTURE INSPECTION (模型结构透视)")
    print("="*60)

    # 检查是否是 Swin 结构
    if not hasattr(model, 'layers'):
        print("❌ 无法解析结构：未找到 model.layers 属性")
        return

    total_esc_count = 0
    total_swin_count = 0

    for i, layer in enumerate(model.layers):
        depth = layer.depth
        res = layer.input_resolution
        print(f"▼ Stage {i} (Resolution: {res} | Depth: {depth})")

        for j, blk in enumerate(layer.blocks):
            block_name = type(blk).__name__

            # 使用不同的图标标记不同类型的 Block
            if "ESC" in block_name:
                marker = "✨ [ESC]"
                total_esc_count += 1
            else:
                marker = "🔹 [Swin]"
                total_swin_count += 1

            # 打印层级
            if j == depth - 1:
                print(f"    └─ Block {j}: {marker} {block_name}")
            else:
                print(f"    ├─ Block {j}: {marker} {block_name}")

        print("") # 空行分隔 Stage

    print("-" * 60)
    print(f"📊 统计结果: Swin Block: {total_swin_count} 个 | ESC Block: {total_esc_count} 个")
    print("="*60 + "\n")


def main():
    print("🚀 开始 ESC 模块集成测试 & 结构验证...")

    # 1. 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/debug_esc.yaml', help='config file path')
    parser.add_argument('--batch-size', type=int, help='batch size')
    parser.add_argument('--data-path', type=str, help='dataset path')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, help='root of output folder')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--optim', type=str, help='optimizer name')

    # 保留参数定义以兼容 get_config
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part')
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, default='O1')

    args = parser.parse_args()

    # 2. 获取配置
    config = get_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")

    # 3. 构建数据加载器
    print(f"⏳ 正在加载数据集: {config.DATA.DATA_PATH} ...")
    try:
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
        print(f"✅ 数据加载成功 (Train: {len(dataset_train)}, Val: {len(dataset_val)})")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 4. 构建模型
    print("⏳ 正在构建模型 (启用 ESC)...")
    try:
        model = build_model(config)
        model.to(device)
        print("✅ 模型构建成功!")

        # === [核心功能] 打印模型结构 ===
        print_model_hierarchy(model)
        # =============================

    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 定义 Loss 和 Optimizer
    criterion = nn.CrossEntropyLoss()
    if config.MODEL.LABEL_SMOOTHING > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.MODEL.LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 6. 运行训练循环 (Smoke Test)
    print("🏃 开始模拟训练循环 (Running 2 batches)...")
    model.train()

    try:
        for idx, (samples, targets) in enumerate(data_loader_train):
            samples = samples.to(device)
            targets = targets.to(device)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            print(f"   Batch {idx+1}: Input {samples.shape} -> ", end="")

            outputs = model(samples)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f} (OK)")

            if idx >= 1:
                break

        print("\n✨ 恭喜！训练流程与结构验证全部通过。")

    except Exception as e:
        print(f"\n❌ 训练循环崩溃: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()