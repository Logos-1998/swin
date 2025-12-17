# Swin-Transformer/models/build.py

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

# [CLEANUP] 只保留 V2 的引用
from .swin_transformer_v2 import SwinTransformerV2
# from .swin_transformer import SwinTransformer
# from .swin_transformer_moe import SwinTransformerMoE
# from .swin_mlp import SwinMLP
# from .simmim import build_simmim

def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    use_fused_layernorm = getattr(config, 'FUSED_LAYERNORM', False)
    if use_fused_layernorm:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    # [CLEANUP] 移除了 SimMIM 的预训练逻辑
    if is_pretrain:
        raise NotImplementedError("SimMIM pretraining is not supported in this cleaned version.")

    # [CLEANUP] 移除了 swin (v1), swin_moe, swin_mlp 的逻辑，只保留 swinv2
    if model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES,
                                  use_esc=config.MODEL.SWINV2.USE_ESC,
                                  esc_pdim=config.MODEL.SWINV2.ESC_PDIM,
                                  esc_kernel_size=config.MODEL.SWINV2.ESC_KERNEL_SIZE)
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model