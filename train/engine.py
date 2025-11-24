import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from data.datasets import build_dataloaders
from models.factory import build_model
from utils.logging import init_logging, log_metrics
from utils.time import format_time
from .early_stopping import EarlyStopping
from .optim import build_optimizer
from .schedulers import build_scheduler, check_batch_size_update


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train(cfg):
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = build_dataloaders(cfg)

    pretrained = cfg.get("pretrained", False)
    model = build_model(cfg["model_name"], num_classes, pretrained).to(device)

    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)
    early_stopper = EarlyStopping(patience=cfg.get("early_patience", 10))

    use_amp = cfg.get("use_amp", False)
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Status: AMP ENABLED")
    else:
        print("Status: AMP DISABLED")

    logger = init_logging(cfg)
    num_epochs = cfg["num_epochs"]
    total_start = time.perf_counter()

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()

        new_bs = check_batch_size_update(cfg, epoch)
        if new_bs:
            print(f"Epoch {epoch}: Batch Size Scheduler triggered. New BS: {new_bs}")
            cfg["batch_size"] = new_bs
            train_loader, val_loader, _ = build_dataloaders(cfg)

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['num_epochs']} [train]", leave=True)

        for x, y in train_iter:
            x, y = x.to(device), y.to(device)

            if cfg.get("optimizer") == "sam":
                if use_amp:
                    with autocast('cuda'):
                        logits = model(x)
                        loss = F.cross_entropy(logits, y)
                    scaler.scale(loss).backward()
                    optimizer.first_step(zero_grad=True)
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                if use_amp:
                    with autocast('cuda'):
                        logits = model(x)
                        loss_2 = F.cross_entropy(logits, y)

                    loss_2.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    logits = model(x)
                    loss_2 = F.cross_entropy(logits, y)
                    loss_2.backward()
                    optimizer.second_step(zero_grad=True)

                loss = loss_2
            else:
                optimizer.zero_grad()
                if use_amp:
                    with autocast('cuda'):
                        logits = model(x)
                        loss = F.cross_entropy(logits, y)

                    if torch.isnan(loss):
                        print("!!! CRITICAL ERROR: Loss is NaN !!!")
                        continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)

                    if torch.isnan(loss):
                        print("!!! CRITICAL ERROR: Loss is NaN !!!")
                        return

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            current_acc = correct / total if total > 0 else 0.0
            train_iter.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{current_acc:.4f}",
            })

        if total > 0:
            train_loss = total_loss / total
            train_acc = correct / total
        else:
            train_loss = 0.0
            train_acc = 0.0

        val_loss, val_acc = evaluate(model, val_loader, device)

        epoch_time = time.perf_counter() - epoch_start
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = remaining_epochs * epoch_time

        tqdm.write(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s | "
            f"ETA: ~{format_time(eta_seconds)}"
        )
        log_metrics(logger, epoch, train_loss, train_acc, val_loss, val_acc)

        if scheduler is not None:
            if cfg.get("scheduler").lower() == "reducelronplateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if early_stopper(val_loss, model):
            tqdm.write("Early stopping.")
            break

    total_time = time.perf_counter() - total_start
    print(f"Training finished in {format_time(total_time)}.")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    if total == 0: return 0.0, 0.0

    loss = total_loss / total
    acc = correct / total
    return loss, acc
