#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from src.scene_classifier.data import build_transforms
from src.scene_classifier.models import build_model
from src.scene_classifier.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    class_names = checkpoint["class_names"]
    _, transform = build_transforms(config["data"])
    model = build_model(config["model"], len(class_names), load_pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    paths = sorted(
        path for path in args.input_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rows = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                inputs = transform(image.convert("RGB")).unsqueeze(0).to(device)
            probabilities = model(inputs).softmax(dim=1)[0]
            confidence, label = probabilities.max(dim=0)
            rows.append(
                {
                    "filename": str(path.relative_to(args.input_dir)),
                    "predicted_label": int(label),
                    "predicted_class": class_names[int(label)],
                    "confidence": float(confidence),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
