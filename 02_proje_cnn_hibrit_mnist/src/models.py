from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet18


class C1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 6, kernel_size=5)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class C2(nn.Module):
    def __init__(self, use_batch_norm: bool = False, dropout_p: float = 0.0) -> None:
        super().__init__()
        layers: list[tuple[str, nn.Module]] = [
            ("conv2", nn.Conv2d(6, 16, kernel_size=5)),
        ]
        if use_batch_norm:
            layers.append(("bn2", nn.BatchNorm2d(16)))
        layers.extend(
            [
                ("relu2", nn.ReLU()),
                ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),
            ]
        )
        if dropout_p > 0:
            layers.append(("drop2", nn.Dropout2d(p=dropout_p)))
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class C3(nn.Module):
    def __init__(self, use_batch_norm: bool = False, dropout_p: float = 0.0) -> None:
        super().__init__()
        layers: list[tuple[str, nn.Module]] = [
            ("conv3", nn.Conv2d(16, 120, kernel_size=5)),
        ]
        if use_batch_norm:
            layers.append(("bn3", nn.BatchNorm2d(120)))
        layers.append(("relu3", nn.ReLU()))
        if dropout_p > 0:
            layers.append(("drop3", nn.Dropout(p=dropout_p)))
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class F4(nn.Module):
    def __init__(self, dropout_p: float = 0.0) -> None:
        super().__init__()
        layers: list[tuple[str, nn.Module]] = [
            ("fc4", nn.Linear(120, 84)),
            ("relu4", nn.ReLU()),
        ]
        if dropout_p > 0:
            layers.append(("drop4", nn.Dropout(p=dropout_p)))
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class F5(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LeNet5Baseline(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5(num_classes=num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        residual_a = self.c2_1(x)
        residual_b = self.c2_2(x)
        x = residual_a + residual_b
        x = self.c3(x)
        x = torch.flatten(x, start_dim=1)
        return self.f4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f5(self.forward_features(x))


class LeNet5Improved(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.c1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 6, kernel_size=5)),
                    ("bn1", nn.BatchNorm2d(6)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("drop1", nn.Dropout2d(p=0.10)),
                ]
            )
        )
        self.c2_1 = C2(use_batch_norm=True, dropout_p=0.15)
        self.c2_2 = C2(use_batch_norm=True, dropout_p=0.15)
        self.c3 = C3(use_batch_norm=True, dropout_p=0.20)
        self.f4 = F4(dropout_p=0.30)
        self.f5 = F5(num_classes=num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        residual_a = self.c2_1(x)
        residual_b = self.c2_2(x)
        x = residual_a + residual_b
        x = self.c3(x)
        x = torch.flatten(x, start_dim=1)
        return self.f4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f5(self.forward_features(x))


class ResNet18ForMNIST(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(self.forward_features(x))
