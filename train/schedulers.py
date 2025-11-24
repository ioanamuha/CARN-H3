from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def build_scheduler(cfg, optimizer):
    scheduler_name = cfg.get("scheduler", "steplr").lower()
    if scheduler_name == "steplr":
        step_size = cfg.get("step_size", 10)
        gamma = cfg.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name == "reducelronplateau":
        patience = cfg.get("patience", 5)
        factor = cfg.get("factor", 0.1)
        return ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    return None


def check_batch_size_update(cfg, epoch):
    # config example:  bs_schedule: {15: 256, 30: 512}
    schedule = cfg.get("bs_schedule", None)
    if schedule and epoch in schedule:
        return schedule[epoch]
    return None
