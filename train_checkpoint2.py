#!/usr/bin/env python
"""Train checkpoint2: regularized RGB ResNet-18 from scratch."""

from pathlib import Path

from train import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "configs" / "checkpoint2.yaml")
