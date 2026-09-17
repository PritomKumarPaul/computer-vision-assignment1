from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class TNet(nn.Module):
    """Exact architecture from the supplied starter notebook."""

    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 15 * 15, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


class SmallCNN(nn.Module):
    """A compact, conventional CNN trained from scratch."""

    def __init__(self, num_classes: int = 16, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_model(model_config: dict, num_classes: int, load_pretrained: bool | None = None):
    name = model_config["name"]
    if name == "tnet":
        return TNet(num_classes=num_classes)
    if name == "small_cnn":
        return SmallCNN(
            num_classes=num_classes,
            dropout=float(model_config.get("dropout", 0.35)),
        )
    if name == "resnet18":
        pretrained = bool(model_config.get("pretrained", True))
        if load_pretrained is not None:
            pretrained = load_pretrained
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        dropout = float(model_config.get("dropout", 0.0))
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model
    raise ValueError(f"Unknown model: {name}")


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    if model_name != "resnet18":
        return
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
