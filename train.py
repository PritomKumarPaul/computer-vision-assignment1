#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.scene_classifier.config import load_config
from src.scene_classifier.complexity import profile_model
from src.scene_classifier.data import build_loaders
from src.scene_classifier.engine import fit
from src.scene_classifier.models import build_model
from src.scene_classifier.utils import (
    count_parameters,
    environment_summary,
    resolve_device,
    set_seed,
    write_json,
)


def build_optimizer(model, training: dict):
    name = training["optimizer"].lower()
    kwargs = {
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, training: dict):
    name = training.get("scheduler", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(training["epochs"])
        )
    raise ValueError(f"Unknown scheduler: {name}")


def append_experiment(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main(default_config: Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=default_config,
        required=default_config is None,
    )
    parser.add_argument("--run-name", help="Override run_name without editing the YAML file.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--num-workers", type=int,
        help="Override the config DataLoader worker count (use 0 in restricted environments).",
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    config = load_config(args.config)
    if args.run_name:
        config["run_name"] = args.run_name
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    set_seed(int(config["seed"]))
    device = resolve_device(args.device)
    train_loader, val_loader, class_names = build_loaders(config, project_root)

    model = build_model(config["model"], len(class_names)).to(device)
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    run_dir = project_root / "artifacts" / "runs" / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    channels = int(config["data"]["channels"])
    image_size = int(config["data"]["image_size"])
    complexity = profile_model(
        model, (1, channels, image_size, image_size), device
    )
    metadata = {
        "config": config,
        "environment": environment_summary(),
        "device": str(device),
        "class_names": class_names,
        "parameters": count_parameters(model),
        "complexity": complexity,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
    }
    write_json(run_dir / "metadata.json", metadata)
    print(f"Run: {config['run_name']} | device: {device}")
    print(
        f"Complexity: {complexity['total_parameters']:,} parameters | "
        f"{complexity['parameter_and_buffer_size_mib']:.3f} MiB parameter storage | "
        f"{complexity['estimated_macs_per_image'] / 1e6:.3f} M MACs/image | "
        f"{complexity['estimated_flops_per_image'] / 1e6:.3f} M FLOPs/image"
    )
    print(f"Train: {metadata['train_samples']} | validation: {metadata['validation_samples']}")

    model, history, best_epoch, best_accuracy, elapsed = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        class_names=class_names,
        device=device,
        run_dir=run_dir,
    )
    result = {
        "run_name": config["run_name"],
        "model": config["model"]["name"],
        "seed": config["seed"],
        "device": str(device),
        "image_size": image_size,
        "batch_size": int(config["data"]["batch_size"]),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "elapsed_seconds": round(elapsed, 3),
        "parameters": metadata["parameters"],
        "trainable_parameters": complexity["trainable_parameters"],
        "model_size_mib": round(complexity["parameter_and_buffer_size_mib"], 6),
        "macs_per_image": complexity["estimated_macs_per_image"],
        "flops_per_image": complexity["estimated_flops_per_image"],
        "average_epoch_seconds": round(
            sum(row["epoch_seconds"] for row in history) / len(history), 3
        ),
        "peak_gpu_memory_mib": max(
            (row["train_peak_gpu_memory_mib"] or 0.0 for row in history),
            default=0.0,
        ),
    }
    write_json(run_dir / "completed.json", result)
    append_experiment(project_root / "results" / "experiments.csv", result)
    print(f"Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
    print(f"Total training time: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print(f"Checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
