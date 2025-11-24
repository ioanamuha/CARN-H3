import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_dataloaders(cfg):
    dataset = cfg["dataset"].lower()
    data_root = cfg.get("data_root", "./data")
    batch_size = cfg.get("batch_size", 128)
    image_size = cfg.get("image_size", 32)
    num_workers = cfg.get("num_workers", 4)

    if dataset == 'mnist':
        train_t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        test_t = train_t
    else:
        train_t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4) if image_size <= 32 else transforms.RandomResizedCrop(
                image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        test_t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    if dataset == "cifar10":
        train_ds = datasets.CIFAR10(root=data_root, train=True, transform=train_t, download=True)
        val_ds = datasets.CIFAR10(root=data_root, train=False, transform=test_t, download=True)
        num_classes = 10
    elif dataset == "cifar100":
        train_ds = datasets.CIFAR100(root=data_root, train=True, transform=train_t, download=True)
        val_ds = datasets.CIFAR100(root=data_root, train=False, transform=test_t, download=True)
        num_classes = 100
    elif dataset == "mnist":
        train_ds = datasets.MNIST(root=data_root, train=True, transform=train_t, download=True)
        val_ds = datasets.MNIST(root=data_root, train=False, transform=test_t, download=True)
        num_classes = 10
    elif dataset == "oxfordiiitpet":
        # 'trainval' is often used for training in this dataset
        train_ds = datasets.OxfordIIITPet(root=data_root, split="trainval", transform=train_t, download=True)
        val_ds = datasets.OxfordIIITPet(root=data_root, split="test", transform=test_t, download=True)
        num_classes = 37
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, num_classes
