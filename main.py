import argparse

from train.engine import train
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Pipeline HW3")

    # --- Config File ---
    parser.add_argument("--config", type=str, default="config_cifar100.yaml", help="Path to YAML config file")

    # --- Data Parameters ---
    parser.add_argument("--dataset", type=str, help="Override dataset (mnist, cifar10, cifar100, oxfordiiitpet)")
    parser.add_argument("--data_root", type=str, help="Override data root path")
    parser.add_argument("--num_classes", type=int, help="Override number of classes")
    parser.add_argument("--image_size", type=int, help="Override image size")
    parser.add_argument("--num_workers", type=int, help="Override number of workers")

    parser.add_argument("--model_name", type=str, help="Override model name (resnet18, resnet50, resnest26d, mlp)")
    parser.add_argument('--pretrained', action='store_true', help='Flag to use pretrained model weights')

    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")

    # --- Optimization Parameters ---
    parser.add_argument("--optimizer", type=str, help="Override optimizer (adam, sgd, adamw, sam, muon)")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, help="Override weight decay")
    parser.add_argument("--momentum", type=float, help="Override momentum")

    # --- Scheduler Parameters ---
    parser.add_argument("--scheduler", type=str, help="Override scheduler (steplr, reducelronplateau)")
    parser.add_argument("--step_size", type=int, help="Override step_size for StepLR")
    parser.add_argument("--gamma", type=float, help="Override gamma for StepLR")
    parser.add_argument("--patience", type=int, help="Override patience for ReduceLROnPlateau")
    parser.add_argument("--factor", type=float, help="Override factor for ReduceLROnPlateau")

    # --- Logging & Efficiency ---
    parser.add_argument("--log_dir", type=str, help="Override log directory")
    parser.add_argument("--use_wandb", action='store_true', help="Flag to enable WandB")
    parser.add_argument("--use_amp", action='store_true', help="Flag to enable AMP (Mixed Precision)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading config from: {args.config}")
    cfg = load_config(args.config)
    print(cfg)

    for arg, value in vars(args).items():
        if arg == "config":
            continue

        if arg in ["pretrained", "use_wandb", "use_amp"]:
            if value:
                print(f"[Override] {arg}: {cfg.get(arg)} -> {value}")
                cfg[arg] = value
        elif value is not None:
            print(f"[Override] {arg}: {cfg.get(arg)} -> {value}")
            cfg[arg] = value

    train(cfg)
