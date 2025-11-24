import wandb
from torch.utils.tensorboard import SummaryWriter


def init_logging(cfg):
    writer = SummaryWriter(log_dir=cfg["log_dir"])
    if cfg.get("use_wandb", False):
        wandb.init(project=cfg["wandb_project"], config=cfg)
    return {"tb": writer, "wandb": wandb if cfg.get("use_wandb", False) else None}


def log_metrics(logger, epoch, train_loss, train_acc, val_loss, val_acc):
    tb = logger["tb"]
    tb.add_scalar("Loss/train", train_loss, epoch)
    tb.add_scalar("Loss/val", val_loss, epoch)
    tb.add_scalar("Acc/train", train_acc, epoch)
    tb.add_scalar("Acc/val", val_acc, epoch)

    if logger["wandb"] is not None:
        logger["wandb"].log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })
