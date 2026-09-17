#!/usr/bin/env python
"""Fine-tune the ImageNet-pretrained ResNet-18 candidate."""

from pathlib import Path

from train import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "configs" / "resnet18.yaml")
