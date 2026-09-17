#!/usr/bin/env python
"""Compare the ResNeXt initialization/freezing strategies in checkpoint 4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


LABELS = {
    "scratch_all": "Scratch: all layers",
    "pretrained_all": "Pretrained: all layers",
    "pretrained_last_stage": "Pretrained: layer4 + head",
    "pretrained_head_only": "Pretrained: head only",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def peak_memory(history_path: Path) -> float:
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [
        float(row["train_peak_gpu_memory_mib"])
        for row in rows
        if row.get("train_peak_gpu_memory_mib") not in {None, "", "None"}
    ]
    return max(values, default=0.0)


def collect(summary_path: Path) -> dict:
    summary = read_json(summary_path)
    config = summary["config"]
    model = config["model"]
    validation = summary["saved_checkpoint_validation"]
    complexity = summary["complexity"]
    variant = summary_path.parent.name
    return {
        "variant": variant,
        "label": LABELS.get(variant, variant),
        "pretrained": bool(model["pretrained"]),
        "trainable_scope": model["trainable_scope"],
        "learning_rate": config["training"]["learning_rate"],
        "best_epoch": summary["best_epoch"],
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "total_parameters": complexity["total_parameters"],
        "trainable_parameters": complexity["trainable_parameters"],
        "trainable_fraction": (
            complexity["trainable_parameters"] / complexity["total_parameters"]
        ),
        "macs_per_image": complexity["estimated_macs_per_image"],
        "training_seconds": summary["final_epoch"]["total_seconds"],
        "peak_gpu_memory_mib": peak_memory(summary_path.parent / "history.csv"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    ranked = sorted(rows, key=lambda row: row["validation_accuracy"], reverse=True)
    lines = [
        "# Checkpoint 4 ResNeXt transfer-learning comparison",
        "",
        "All four runs use ResNeXt-50 32x4d, the same leakage-controlled split, 224x224 RGB inputs, ImageNet normalization, augmentation, seed, batch size, regularization, scheduler, and maximum epoch budget. Learning rates are selected for each optimization scope and are recorded below.",
        "",
        "| Rank | Setup | LR | Val. accuracy | Macro F1 | Trainable / total parameters | Trainable | Train time | Peak GPU |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['label']} | {row['learning_rate']:.1e} | "
            f"{100 * row['validation_accuracy']:.2f}% | "
            f"{100 * row['validation_macro_f1']:.2f}% | "
            f"{row['trainable_parameters'] / 1e6:.3f}M / "
            f"{row['total_parameters'] / 1e6:.3f}M | "
            f"{100 * row['trainable_fraction']:.2f}% | "
            f"{row['training_seconds']:.1f}s | "
            f"{row['peak_gpu_memory_mib']:.1f} MiB |"
        )
    lines.extend(
        [
            "",
            "The convolutional MAC count is nearly identical across setups at inference time. Freezing changes gradient computation, optimizer state, training memory, and training time; it does not remove backbone operations from inference.",
            "",
            "The labeled test set is intentionally excluded from this comparison.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_plot(path: Path, rows: list[dict]) -> None:
    ranked = sorted(rows, key=lambda row: row["validation_accuracy"], reverse=True)
    labels = [row["label"] for row in ranked]
    accuracies = [row["validation_accuracy"] for row in ranked]
    trainable = [row["trainable_parameters"] for row in ranked]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(labels[::-1], accuracies[::-1])
    axes[0].set(xlabel="Validation accuracy", title="Validation ranking", xlim=(0, 1))
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(labels[::-1], trainable[::-1], color="tab:orange")
    axes[1].set_xscale("log")
    axes[1].set(
        xlabel="Trainable parameters (log scale)",
        title="Optimization footprint",
    )
    axes[1].grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="checkpoint4")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    suite_root = project_root / "results" / args.name
    summaries = sorted(suite_root.glob("*/summary.json"))
    if not summaries:
        raise SystemExit(f"No finalized variants found beneath {suite_root}")

    rows = [collect(summary_path) for summary_path in summaries]
    write_csv(suite_root / "comparison.csv", rows)
    write_markdown(suite_root / "comparison.md", rows)
    figure_dir = project_root / "reports" / "figures" / args.name
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_plot(figure_dir / "transfer_comparison.png", rows)
    print(f"Compared {len(rows)} transfer-learning setups")
    print(f"  table: {suite_root / 'comparison.csv'}")
    print(f"  report: {suite_root / 'comparison.md'}")
    print(f"  figure: {figure_dir / 'transfer_comparison.png'}")


if __name__ == "__main__":
    main()
