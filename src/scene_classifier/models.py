from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNeXt50_32X4D_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet18,
    resnext50_32x4d,
)


RESNEXT_SCOPES = {"all", "layer4_and_head", "head_only"}
TRANSFER_SCOPES = {"all", "last_stage_and_head", "head_only"}


def _configure_resnext_trainable_scope(model: nn.Module, scope: str) -> None:
    """Freeze the requested ResNeXt blocks and record modules kept in eval mode."""
    if scope not in RESNEXT_SCOPES:
        raise ValueError(
            f"Unknown ResNeXt trainable_scope: {scope!r}; "
            f"expected one of {sorted(RESNEXT_SCOPES)}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = scope == "all"

    frozen_modules: tuple[str, ...] = ()
    if scope == "layer4_and_head":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        frozen_modules = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3")
    elif scope == "head_only":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        frozen_modules = (
            "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3",
            "layer4", "avgpool",
        )

    # engine.train_one_epoch uses these names to prevent frozen BatchNorm
    # running statistics from changing after model.train() is called.
    model._frozen_module_names = frozen_modules


def _configure_transfer_scope(model: nn.Module, architecture: str, scope: str) -> None:
    """Apply comparable architecture-specific freezing policies."""
    if scope not in TRANSFER_SCOPES:
        raise ValueError(
            f"Unknown transfer trainable_scope: {scope!r}; "
            f"expected one of {sorted(TRANSFER_SCOPES)}"
        )

    policies = {
        "resnet18": {
            "last_stage_and_head": (
                ("layer4", "fc"),
                ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"),
            ),
            "head_only": (
                ("fc",),
                (
                    "conv1", "bn1", "relu", "maxpool", "layer1", "layer2",
                    "layer3", "layer4", "avgpool",
                ),
            ),
        },
        "densenet121": {
            "last_stage_and_head": (
                ("features.denseblock4", "features.norm5", "classifier"),
                (
                    "features.conv0", "features.norm0", "features.relu0",
                    "features.pool0", "features.denseblock1", "features.transition1",
                    "features.denseblock2", "features.transition2",
                    "features.denseblock3", "features.transition3",
                ),
            ),
            "head_only": (("classifier",), ("features",)),
        },
        "efficientnet_b0": {
            "last_stage_and_head": (
                ("features.7", "features.8", "classifier"),
                tuple(f"features.{index}" for index in range(7)),
            ),
            "head_only": (("classifier",), ("features", "avgpool")),
        },
        "convnext_tiny": {
            "last_stage_and_head": (
                ("features.7", "classifier"),
                tuple(f"features.{index}" for index in range(7)),
            ),
            # ConvNeXt's original classifier contains a LayerNorm. Keeping only
            # classifier.2 trainable means the newly inserted Dropout/Linear
            # head is optimized while the pretrained normalization is frozen.
            "head_only": (
                ("classifier.2",),
                ("features", "avgpool", "classifier.0"),
            ),
        },
    }
    if architecture not in policies:
        raise ValueError(f"Unsupported transfer architecture: {architecture}")

    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
        model._frozen_module_names = ()
        return

    trainable_modules, frozen_modules = policies[architecture][scope]
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in trainable_modules:
        for parameter in model.get_submodule(module_name).parameters():
            parameter.requires_grad = True
    model._frozen_module_names = frozen_modules


def _should_load_pretrained(model_config: dict, load_pretrained: bool | None) -> bool:
    requested = bool(model_config.get("pretrained", False))
    return requested if load_pretrained is None else requested and load_pretrained


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
    if name == "resnext50_32x4d_transfer":
        should_load_pretrained = _should_load_pretrained(
            model_config, load_pretrained
        )
        weights_name = model_config.get("weights", "IMAGENET1K_V2")
        if weights_name != "IMAGENET1K_V2":
            raise ValueError(
                "resnext50_32x4d_transfer currently supports weights: IMAGENET1K_V2"
            )
        weights = (
            ResNeXt50_32X4D_Weights.IMAGENET1K_V2
            if should_load_pretrained
            else None
        )
        model = resnext50_32x4d(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(float(model_config.get("dropout", 0.3))),
            nn.Linear(model.fc.in_features, num_classes),
        )
        _configure_resnext_trainable_scope(
            model, model_config.get("trainable_scope", "all")
        )
        return model
    if name in {
        "resnet18_transfer",
        "densenet121_transfer",
        "efficientnet_b0_transfer",
        "convnext_tiny_transfer",
    }:
        architecture = name.removesuffix("_transfer")
        scope = model_config.get("trainable_scope", "all")
        dropout = float(model_config.get("dropout", 0.3))
        should_load = _should_load_pretrained(model_config, load_pretrained)

        if architecture == "resnet18":
            expected_weights = "IMAGENET1K_V1"
            weights = ResNet18_Weights.IMAGENET1K_V1 if should_load else None
            model = resnet18(weights=weights)
            model.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes)
            )
        elif architecture == "densenet121":
            expected_weights = "IMAGENET1K_V1"
            weights = DenseNet121_Weights.IMAGENET1K_V1 if should_load else None
            model = densenet121(weights=weights)
            model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.classifier.in_features, num_classes),
            )
        elif architecture == "efficientnet_b0":
            expected_weights = "IMAGENET1K_V1"
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if should_load else None
            model = efficientnet_b0(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        else:
            expected_weights = "IMAGENET1K_V1"
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if should_load else None
            model = convnext_tiny(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )

        if model_config.get("weights", expected_weights) != expected_weights:
            raise ValueError(
                f"{name} currently supports weights: {expected_weights}"
            )
        _configure_transfer_scope(model, architecture, scope)
        return model
    raise ValueError(f"Unknown model: {name}")
