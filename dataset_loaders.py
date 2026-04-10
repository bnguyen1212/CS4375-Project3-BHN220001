"""Dataset loading helpers for MNIST and CIFAR-10 using PyTorch/torchvision.

Preprocessing summary:
- Convert images to PyTorch tensors with torchvision.transforms.ToTensor().
- ToTensor() rescales pixel values from [0, 255] to [0, 1].
    - MNIST uses a single grayscale channel; CIFAR-10 uses three RGB channels.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def _build_transform():
    return transforms.Compose([transforms.ToTensor()])


def load_mnist(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    split_seed: int = 42,
):
    """Load MNIST train/val/test datasets and return DataLoader objects.

    Args:
        data_dir: Directory where the dataset will be stored.
        batch_size: Number of samples per batch.
        num_workers: DataLoader worker count.
        pin_memory: If None, defaults to torch.cuda.is_available().
        split_seed: Random seed for deterministic train/validation split.

    Returns:
        train_loader, val_loader, test_loader
    """

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    transform = _build_transform()

    train_dataset_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = random_split(train_dataset_full, [50000, 10000], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def load_cifar10(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    split_seed: int = 42,
):
    """Load CIFAR-10 train/val/test datasets and return DataLoader objects.

    Args:
        data_dir: Directory where the dataset will be stored.
        batch_size: Number of samples per batch.
        num_workers: DataLoader worker count.
        pin_memory: If None, defaults to torch.cuda.is_available().
        split_seed: Random seed for deterministic train/validation split.

    Returns:
        train_loader, val_loader, test_loader
    """

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    transform = _build_transform()

    train_dataset_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = random_split(train_dataset_full, [45000, 5000], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    mnist_train, mnist_val, mnist_test = load_mnist()
    cifar_train, cifar_val, cifar_test = load_cifar10()

    print(f"MNIST train batches: {len(mnist_train)}")
    print(f"MNIST val batches: {len(mnist_val)}")
    print(f"MNIST test batches: {len(mnist_test)}")
    print(f"CIFAR-10 train batches: {len(cifar_train)}")
    print(f"CIFAR-10 val batches: {len(cifar_val)}")
    print(f"CIFAR-10 test batches: {len(cifar_test)}")
