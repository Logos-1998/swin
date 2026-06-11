import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib
# [终极防崩溃]: 彻底封印 Matplotlib 的 GUI 内存泄漏，强制后台纯净渲染
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import importlib

from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# === 用户配置区 (请核对以下绝对路径) ===
# ==========================================

INPUT_IMAGE_DIR = r"D:\Documents\Swin-Transformer\dataset\val\OPA"
OUTPUT_IMAGE_DIR = r"D:\Documents\Swin-Transformer\results\cam\OPA"

COMP_DIR = r"D:\Documents\Comparison"
SWIN_DIR = r"D:\Documents\Swin-Transformer"
SWIN_CFG = r"D:\Documents\Swin-Transformer\configs\exp4_full.yaml"
CLINICAL_CSV_PATH = r"D:\Documents\Swin-Transformer\dataset\clinical_data.csv"

NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = 256

MEAN = [0.1189937, 0.1189937, 0.1189937]
STD = [0.1225612, 0.1225612, 0.1225612]

MODELS_TO_TEST = {
    "ResNet50": ("resnet50", r"D:\Documents\Comparison\results\resnet50\resnet50_best.pth"),
    "DenseNet121": ("densenet121", r"D:\Documents\Comparison\results\densenet121\densenet121_best.pth"),
    "ViT-B/16": ("vit_b_16", r"D:\Documents\Comparison\results\vit_b_16\vit_b_16_best.pth"),
    "EfficientNetV2-S": ("efficientnet_v2_s", r"D:\Documents\Comparison\results\efficientnet_v2_s\efficientnet_v2_s_best.pth"),
    "ResNet501": ("resnet50", r"D:\Documents\Comparison\results\resnet50_2\resnet50_best.pth"),

    # "EM-SwinT": ("custom_swin", r"D:\Documents\Swin-Transformer\output\exp4_full_model\checkpoint_best.pth")
}

# ==========================================
# === 自动化数据处理模块 (全局预计算) ===
# ==========================================

def build_clinical_dict(csv_path):
    print(f">>> 正在读取并进行全表临床特征标准化...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到临床数据表格: {csv_path}")

    df = pd.read_csv(csv_path)
    target_features = ['age', 'gender', 'height', 'weight']
    clinical_dict = {}

    for col in target_features:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
                mean_val, std_val = df[col].mean(), df[col].std()
                if std_val > 1e-6:
                    df[col] = (df[col] - mean_val) / std_val
            else:
                df[col] = df[col].astype('category').cat.codes
        else:
            df[col] = 0.0

    for idx, row in df.iterrows():
        basename = os.path.basename(str(row['image_path']).replace('\\', '/'))
        feats = row[target_features].values.astype(np.float32)
        clinical_dict[basename] = [round(float(x), 4) for x in feats]

    print(f"✅ 成功提取并标准化 {len(clinical_dict)} 条患者临床记录。\n")
    return clinical_dict

# ==========================================
# === 核心逻辑：模块环境隔离与加载 ===
# ==========================================

CONFLICTING_MODULES = ['config', 'utils', 'model_factory', 'models', 'models.build', 'main', 'evaluate']

def clean_module_cache():
    for mod in CONFLICTING_MODULES:
        sys.modules.pop(mod, None)

class SwinFusionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.use_fusion = getattr(model, 'use_fusion', False)
        self.clinical_data = None

    def forward(self, x):
        if self.use_fusion:
            if self.clinical_data is not None:
                clin_tensor = torch.tensor([self.clinical_data], dtype=torch.float32, device=x.device)
                clin_tensor = clin_tensor.expand(x.size(0), -1)
            else:
                try:
                    clin_layer = self.model.fusion_head.clin_proj
                    in_dim = clin_layer[0].in_features if isinstance(clin_layer, torch.nn.Sequential) else clin_layer.in_features
                except Exception:
                    in_dim = 16
                clin_tensor = torch.zeros((x.size(0), in_dim), device=x.device)
            return self.model(x, clinical_data=clin_tensor)
        return self.model(x)

def load_comparison_model(arch_name, weight_path):
    clean_module_cache()
    original_path = list(sys.path)
    sys.path.insert(0, COMP_DIR)

    try:
        model_factory = importlib.import_module('model_factory')
        model = model_factory.get_model(arch_name.lower(), NUM_CLASSES)
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=False))
    finally:
        sys.path = original_path
        clean_module_cache()
    return model

def load_custom_swin(weight_path):
    clean_module_cache()
    original_path = list(sys.path)
    sys.path.insert(0, SWIN_DIR)

    try:
        build_module = importlib.import_module('models.build')
        config_module = importlib.import_module('config')

        class MockArgs:
            cfg = SWIN_CFG
            batch_size = None; data_path = None; zip = False; cache_mode = None
            resume = None; accumulation_steps = None; use_checkpoint = False
            amp_opt_level = None; output = None; tag = None; eval = False
            throughput = False; local_rank = 0

        args = MockArgs()
        config = config_module.get_config(args)

        model = build_module.build_model(config)
        if os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    finally:
        sys.path = original_path
        clean_module_cache()

    return SwinFusionWrapper(model)

def swin_reshape_transform(tensor):
    B, L, C = tensor.shape
    H = W = int(np.sqrt(L))
    result = tensor.reshape(B, H, W, C)
    return result.transpose(2, 3).transpose(1, 2)

def torchvision_vit_reshape_transform(tensor):
    result = tensor[:, 1:, :]
    B, L, C = result.shape
    H = W = int(np.sqrt(L))
    result = result.reshape(B, H, W, C)
    return result.transpose(2, 3).transpose(1, 2)

def get_cam_target_and_reshape(model, arch_name):
    arch_lower = arch_name.lower()
    if "resnet" in arch_lower:
        return [model.layer4[-1]], None
    elif "densenet" in arch_lower:
        return [model.features[-1]], None
    elif "efficientnet" in arch_lower:
        return [model.features[-1]], None
    elif "convnext" in arch_lower:
        return [model.features[-1][-1]], None
    elif "vit" in arch_lower:
        return [model.encoder.layers[-1].ln_1], torchvision_vit_reshape_transform
    elif "custom_swin" in arch_lower:
        real_model = model.model if hasattr(model, 'use_fusion') else model
        return [real_model.norm], swin_reshape_transform
    else:
        return [list(model.children())[-2]], None

# ==========================================
# === 主绘图逻辑 (全驻留流式流处理架构) ===
# ==========================================

def main():
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    # 1. 预构建临床特征字典
    clinical_dict = build_clinical_dict(CLINICAL_CSV_PATH)

    # 2. 递归搜寻图像
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    image_paths = []
    for root, dirs, files in os.walk(INPUT_IMAGE_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"❌ 错误: 未找到任何图像文件。")
        return

    print(f"✅ 共找到 {len(image_paths)} 张图像。\n")

    # 3. [架构核心优化]：一次性把所有模型安全地装进显存！
    print(">>> 正在将所有模型一次性常驻加载至 GPU (耗时约十几秒，请稍候)...")
    loaded_models = {}
    for display_title, (arch_name, weight_path) in MODELS_TO_TEST.items():
        if "custom_swin" in arch_name.lower():
            model = load_custom_swin(weight_path)
        else:
            model = load_comparison_model(arch_name, weight_path)

        model.to(DEVICE)
        model.eval()

        target_layers, reshape_fn = get_cam_target_and_reshape(model, arch_name)

        # 将模型及其配套工具存入字典
        loaded_models[display_title] = {
            'model': model,
            'arch_name': arch_name,
            'target_layers': target_layers,
            'reshape_fn': reshape_fn
        }
    print("✅ 所有模型均已成功驻留 GPU！开始极速出图流水线...\n")

    preprocess = transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # 4. 流式处理：1图进 -> 瞬过5模型 -> 存图 -> 清理内存 (严格的 O(1) 内存)
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)

        # A. 读图预处理
        raw_pil = Image.open(img_path).convert('RGB')
        vis_img_base = cv2.resize(np.array(raw_pil), (TARGET_SIZE, TARGET_SIZE))
        vis_img_float = np.float32(vis_img_base) / 255.0
        input_tensor_gpu = preprocess(raw_pil).unsqueeze(0).to(DEVICE)

        feats = clinical_dict.get(img_name, [0.0, 0.0, 0.0, 0.0])

        plots = [vis_img_base]
        titles = ["Original Image"]

        # B. 秒过 5 个驻留模型
        for display_title, m_data in loaded_models.items():
            model = m_data['model']

            # Swin 动态注入特征
            if "custom_swin" in m_data['arch_name'].lower():
                model.clinical_data = feats

            try:
                cam = LayerCAM(model=model, target_layers=m_data['target_layers'], reshape_transform=m_data['reshape_fn'])
                grayscale_cam = cam(input_tensor=input_tensor_gpu, targets=None)[0, :]
                cam_image = show_cam_on_image(vis_img_float, grayscale_cam, use_rgb=True)

                plots.append(cam_image)
                titles.append(display_title)
            except Exception as e:
                plots.append(np.zeros_like(vis_img_base))
                titles.append(f"{display_title}(Error)")

        # C. 画图、保存、强制销毁当前画布
        num_cols = len(plots)
        fig, axes = plt.subplots(1, num_cols, figsize=(4.5 * num_cols, 4.5))
        if num_cols == 1: axes = [axes]

        for ax, img_arr, title in zip(axes, plots, titles):
            ax.imshow(img_arr)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_IMAGE_DIR, f"cam_{img_name}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)

        # 这三步彻底扼杀内存泄漏
        plt.clf()
        plt.close(fig)
        del raw_pil, vis_img_base, vis_img_float, input_tensor_gpu, plots, titles

        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_paths):
            print(f"   [{idx + 1}/{len(image_paths)}] 完成保存: {img_name}")

    print(f"\n🎉 批量流式测试圆满完成！无任何内存泄漏。所有图像已保存至:\n   {OUTPUT_IMAGE_DIR}")

if __name__ == "__main__":
    main()