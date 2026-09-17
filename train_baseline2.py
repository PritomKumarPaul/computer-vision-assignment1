#!/usr/bin/env python
"""Train the compact CNN-from-scratch baseline."""

from pathlib import Path

from train import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "configs" / "small_cnn.yaml")
