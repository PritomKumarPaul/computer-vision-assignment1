#!/usr/bin/env python
"""Compare the ConvNeXt-Large candidate with the ConvNeXt-Tiny winner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    (
        "ConvNeXt-Tiny (checkpoint 5)",
        ROOT / "results/checkpoint5/convnext_tiny_pretrained_all/summary.json",
    ),
    (
        "ConvNeXt-Large (checkpoint 6)",
        ROOT / "results/checkpoint6/convnext_large_pretrained_all/summary.json",
    ),
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def peak_training_memory(history_path: Path) -> float:
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    return max(
        float(row["train_peak_gpu_memory_mib"])
        for row in history
        if row.get("train_peak_gpu_memory_mib") not in {None, "", "None"}
    )


def collect(label: str, summary_path: Path) -> dict:
    if not summary_path.is_file():
        raise SystemExit(f"Missing finalized summary: {summary_path}")
    summary = read_json(summary_path)
    metrics = summary["saved_checkpoint_validation"]
    complexity = summary["complexity"]
    return {
        "model": label,
        "validation_accuracy": metrics["accuracy"],
        "validation_macro_f1": metrics["macro_f1"],
        "best_epoch": summary["best_epoch"],
        "epochs_completed": summary["epochs_completed"],
        "parameters": complexity["total_parameters"],
        "model_size_mib": complexity["parameter_and_buffer_size_mib"],
        "macs_per_image": complexity["estimated_macs_per_image"],
        "training_seconds": summary["final_epoch"]["total_seconds"],
        "peak_gpu_memory_mib": peak_training_memory(
            summary_path.parent / "history.csv"
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Checkpoint 6 ConvNeXt capacity comparison",
        "",
        "Both models use ImageNet-1K initialization, full fine-tuning, the same leakage-controlled split, 224x224 RGB inputs, augmentation, optimizer, learning rate, regularization, scheduler, batch size, and model-selection rule.",
        "",
        "| Model | Val. accuracy | Macro F1 | Parameters | MACs/image | Peak GPU | Train time | Best epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {100 * row['validation_accuracy']:.2f}% | "
            f"{100 * row['validation_macro_f1']:.2f}% | "
            f"{row['parameters'] / 1e6:.2f}M | "
            f"{row['macs_per_image'] / 1e9:.2f}G | "
            f"{row['peak_gpu_memory_mib']:.1f} MiB | "
            f"{row['training_seconds']:.1f}s | {row['best_epoch']} |"
        )
    lines.extend(
        [
            "",
            "The labeled test set is intentionally excluded. Select the final model using validation evidence before evaluating the test set once.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_plot(path: Path, rows: list[dict]) -> None:
    labels = [row["model"].split(" (")[0] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].bar(labels, [row["validation_accuracy"] for row in rows])
    axes[0].set(title="Validation accuracy", ylim=(0, 1), ylabel="Accuracy")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].bar(labels, [row["macs_per_image"] / 1e9 for row in rows], color="tab:orange")
    axes[1].set(title="Inference compute", ylabel="GMACs/image")
    axes[2].bar(labels, [row["peak_gpu_memory_mib"] for row in rows], color="tab:green")
    axes[2].set(title="Training memory", ylabel="Peak allocated MiB")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    rows = [collect(label, path) for label, path in CANDIDATES]
    result_dir = ROOT / "results/checkpoint6"
    figure_dir = ROOT / "reports/figures/checkpoint6"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result_dir / "comparison.csv", rows)
    write_markdown(result_dir / "comparison.md", rows)
    save_plot(figure_dir / "capacity_comparison.png", rows)
    print(f"Compared {len(rows)} ConvNeXt capacities")
    print(f"  report: {result_dir / 'comparison.md'}")
    print(f"  figure: {figure_dir / 'capacity_comparison.png'}")


if __name__ == "__main__":
    main()
