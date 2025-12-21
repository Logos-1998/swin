# Swin-Transformer/optimizer.py

from torch import optim as optim

def build_optimizer(config, model):
    """
    构建优化器 (AdamW)
    支持参数分组：排除 bias、norm 以及特定关键字参数的 Weight Decay
    """
    # 1. 获取模型定义的排除列表 (如 absolute_pos_embed)
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    # 2. 执行参数分组逻辑
    # 传入 config 中的 WEIGHT_DECAY 值，避免硬编码
    parameters = set_weight_decay(model, skip, skip_keywords, config.TRAIN.WEIGHT_DECAY)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None

    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer: {opt_lower}")

    return optimizer

def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.05):
    """
    将参数分为需要 decay 和不需要 decay 的两组。
    针对 Swin V2 和 ESC 模块进行特殊保护。
    """
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 忽略冻结的权重

        # 满足以下任一条件则不进行 weight decay:
        # 1. 维度为 1 (如 LayerNorm 的 weight/bias, 全连接层的 bias)
        # 2. 名称显式以 ".bias" 结尾
        # 3. 名称在模型的 skip_list 中
        # 4. 名称包含特定的排除关键字 (如 logit_scale, esc_plk_filter, cpb_mlp)
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [
        {'params': has_decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.}
    ]

def check_keywords_in_name(name, keywords=()):
    """检查参数名中是否包含特定的排除关键字"""
    for keyword in keywords:
        if keyword in name:
            return True
    return False