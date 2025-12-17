# Swin-Transformer/optimizer.py

from torch import optim as optim

def build_optimizer(config, model):
    """
    构建优化器 (AdamW)
    支持 Layer-wise learning rate decay (Swin 特性)
    """
    # 1. 过滤不需要梯度的参数 (Frozen parameters)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # (注: 如果想要对 ClinicalHead 使用不同的学习率，可以在这里分组)
    # 暂时使用全局统一配置

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

def check_keywords_in_name(name, keywords):
    for keyword in keywords:
        if keyword in name:
            return True
    return False