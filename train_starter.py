#!/usr/bin/env python
"""Train the supplied one-convolution starter CNN."""

from pathlib import Path

from train import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "configs" / "starter_cnn.yaml")
