from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os

from config import TrainingConfig


def get_cifar10_dataloaders(config: TrainingConfig):
    """Get CIFAR10 data loaders.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    # Define transformations (e.g., convert to tensor and normalize)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load training data
    trainset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.get_device().type == "cuda" else False,
    )

    # Load test data
    testset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.get_device().type == "cuda" else False,
    )

    return trainloader, testloader


def get_imagenet_dataloaders(config: TrainingConfig):
    """Get ImageNet data loaders.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # ImageNet normalization values (ImageNet statistics)
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    # Define training transformations
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    # Define validation transformations
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    # Load training data (assumes ImageNet structure: data_root/imagenet/train/)
    train_dir = f"{config.data_root}/imagenet/train"
    os.makedirs(train_dir, exist_ok=True)

    trainset = torchvision.datasets.ImageFolder(
        root=train_dir, transform=train_transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.get_device().type == "cuda" else False,
    )

    # Load validation data (assumes ImageNet structure: data_root/imagenet/val/)
    val_dir = f"{config.data_root}/imagenet/val"
    os.makedirs(val_dir, exist_ok=True)

    valset = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transform)
    valloader = DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.get_device().type == "cuda" else False,
    )

    return trainloader, valloader
