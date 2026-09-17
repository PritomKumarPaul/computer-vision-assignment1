#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.scene_classifier.data import build_loaders, build_test_loader
from src.scene_classifier.engine import evaluate
from src.scene_classifier.models import build_model
from src.scene_classifier.utils import resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--num-workers", type=int,
        help="Override the checkpoint DataLoader worker count.",
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit(
            "Refusing test evaluation during model development. Add "
            "--confirm-final-test only after final model selection."
        )

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    class_names = checkpoint["class_names"]
    model = build_model(config["model"], len(class_names), load_pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    if args.split == "val":
        _, loader, manifest_classes = build_loaders(config, args.project_root.resolve())
        if manifest_classes != class_names:
            raise ValueError("Validation manifest class order differs from checkpoint.")
    else:
        loader = build_test_loader(config, args.project_root.resolve(), class_names)

    metrics = evaluate(model, loader, device, class_names)
    print(f"{args.split} loss: {metrics['loss']:.4f}")
    print(f"{args.split} accuracy: {metrics['accuracy']:.4f}")
    print(f"{args.split} macro-F1: {metrics['macro_f1']:.4f}")
    for name, recall in metrics["per_class_recall"].items():
        print(f"  {name:14s} recall={recall:.4f}")

    output = args.output or args.checkpoint.parent / f"{args.split}_metrics.json"
    write_json(output, metrics)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
