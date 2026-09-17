from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


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
    raise ValueError(f"Unknown model: {name}")
