from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass
class DataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    feature_train_loader: DataLoader
    feature_test_loader: DataLoader
    num_classes: int
    class_names: list[str]


def build_mnist_dataloaders(
    data_root: str = "./data",
    batch_size: int = 64,
    feature_batch_size: int = 256,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DataBundle:
    use_cuda = torch.cuda.is_available()
    lenet_transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    try:
        train_dataset = datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=lenet_transform,
        )
        test_dataset = datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=lenet_transform,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "MNIST veri kumesi indirilemedi veya eksik kaldı. "
            "Ag erisimi saglayin ya da yarim kalmis dosyalari silip tekrar deneyin: "
            f"{data_root}/MNIST/raw"
        ) from error

    if max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_test_samples is not None:
        test_dataset = Subset(test_dataset, range(min(max_test_samples, len(test_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    feature_train_loader = DataLoader(
        train_dataset,
        batch_size=feature_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    feature_test_loader = DataLoader(
        test_dataset,
        batch_size=feature_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    class_names = [str(index) for index in range(10)]

    return DataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        feature_train_loader=feature_train_loader,
        feature_test_loader=feature_test_loader,
        num_classes=10,
        class_names=class_names,
    )
