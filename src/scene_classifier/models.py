from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet18,
    resnext50_32x4d,
)


class StarterCNN(nn.Module):
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


def build_model(model_config: dict, num_classes: int, load_pretrained: bool | None = None):
    del load_pretrained
    name = model_config["name"]
    if name == "starter_cnn":
        return StarterCNN(num_classes=num_classes)
    if name == "resnet18_scratch":
        if bool(model_config.get("pretrained", False)):
            raise ValueError(
                "checkpoint2 is intentionally trained from scratch; "
                "set pretrained: false"
            )
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(float(model_config.get("dropout", 0.3))),
            nn.Linear(in_features, num_classes),
        )
        return model
    if name in {
        "densenet121_scratch",
        "efficientnet_b0_scratch",
        "mobilenet_v3_large_scratch",
        "resnext50_32x4d_scratch",
        "convnext_tiny_scratch",
    }:
        if bool(model_config.get("pretrained", False)):
            raise ValueError("checkpoint3 architecture comparisons use pretrained: false")
        dropout = float(model_config.get("dropout", 0.3))

        if name == "densenet121_scratch":
            model = densenet121(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.classifier.in_features, num_classes),
            )
        elif name == "efficientnet_b0_scratch":
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        elif name == "mobilenet_v3_large_scratch":
            model = mobilenet_v3_large(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-2] = nn.Dropout(dropout)
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif name == "resnext50_32x4d_scratch":
            model = resnext50_32x4d(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes)
            )
        else:
            model = convnext_tiny(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        return model
    raise ValueError(f"Unknown model: {name}")
