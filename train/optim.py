import torch.optim as optim
from .optimizers.sam import SAM
from .optimizers.moun import Muon

def build_optimizer(cfg, params):
    name = cfg.get("optimizer", "rmsprop").lower()

    lr = float(cfg.get("lr", 0.001))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))

    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sam":
        return SAM(params, base_optimizer=optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "muon":
        return Muon(params, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer {name}")
