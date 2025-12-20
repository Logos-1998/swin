# Swin-Transformer/optimizer.py
from torch import optim as optim

def build_optimizer(config, model):
    """
    构建优化器 (AdamW)
    修复：引入参数分组，确保 bias 和 norm 等参数不进行 Weight Decay
    """
    # 1. 获取模型定义的排除列表
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    # 2. 对参数进行分组
    parameters = set_weight_decay(model, skip, skip_keywords)

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

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    """
    将参数分为需要 decay 和不需要 decay 的两组
    参考官方实现逻辑
    """
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 忽略冻结权重

        # 满足以下任一条件则不进行 weight decay:
        # 1. 维度为1 (如 bias, norm.weight, norm.bias)
        # 2. 名称在排除列表中
        # 3. 名称包含特定的排除关键字
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [
        {'params': has_decay, 'weight_decay': 0.05}, # 这里的权重衰减值由配置决定
        {'params': no_decay, 'weight_decay': 0.}      # 强制不衰减组
    ]

def check_keywords_in_name(name, keywords=()):
    for keyword in keywords:
        if keyword in name:
            return True
    return False