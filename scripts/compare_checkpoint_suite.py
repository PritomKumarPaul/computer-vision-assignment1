#!/usr/bin/env python
"""Create tabular and visual comparisons for a checkpoint architecture suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


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


def collect(summary_path: Path, label: str) -> dict:
    summary = read_json(summary_path)
    validation = summary["saved_checkpoint_validation"]
    complexity = summary["complexity"]
    final_epoch = summary["final_epoch"]
    return {
        "architecture": label,
        "model_name": summary["config"]["model"]["name"],
        "best_epoch": summary["best_epoch"],
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "parameters": complexity["total_parameters"],
        "model_size_mib": complexity["parameter_and_buffer_size_mib"],
        "macs_per_image": complexity["estimated_macs_per_image"],
        "flops_per_image": complexity["estimated_flops_per_image"],
        "training_seconds": final_epoch["total_seconds"],
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
        "# Checkpoint 3 architecture comparison",
        "",
        "All checkpoint-3 variants use the same split, RGB augmentation, input size, training budget, optimizer, scheduler, and regularization. Checkpoint 2 is included as the ResNet-18 reference.",
        "",
        "| Rank | Architecture | Val. accuracy | Macro F1 | Parameters | MACs/image | Train time | Peak GPU |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['architecture']} | "
            f"{100 * row['validation_accuracy']:.2f}% | "
            f"{100 * row['validation_macro_f1']:.2f}% | "
            f"{row['parameters'] / 1e6:.2f}M | "
            f"{row['macs_per_image'] / 1e6:.1f}M | "
            f"{row['training_seconds']:.1f}s | "
            f"{row['peak_gpu_memory_mib']:.1f} MiB |"
        )
    lines.extend(
        [
            "",
            "The comparison is a controlled recipe comparison, not an exhaustive hyperparameter search. A model may rank lower because the common optimizer or schedule is less suitable for that architecture.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_plot(path: Path, rows: list[dict]) -> None:
    ranked = sorted(rows, key=lambda row: row["validation_accuracy"])
    labels = [row["architecture"] for row in ranked]
    accuracies = [row["validation_accuracy"] for row in ranked]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(labels, accuracies)
    axes[0].set(xlabel="Validation accuracy", title="Accuracy ranking", xlim=(0, 1))
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(axis="x", alpha=0.25)

    for row in rows:
        axes[1].scatter(
            row["macs_per_image"], row["validation_accuracy"], s=75
        )
        axes[1].annotate(
            row["architecture"],
            (row["macs_per_image"], row["validation_accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xscale("log")
    axes[1].set(
        xlabel="MACs per image (log scale)",
        ylabel="Validation accuracy",
        title="Accuracy–compute tradeoff",
        ylim=(0, 1),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="checkpoint3")
    parser.add_argument(
        "--reference", type=Path, default=Path("results/checkpoint2/summary.json")
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    suite_root = project_root / "results" / args.name
    summaries = sorted(suite_root.glob("*/summary.json"))
    if not summaries:
        raise SystemExit(f"No finalized variants found beneath {suite_root}")

    rows = []
    reference = args.reference
    if not reference.is_absolute():
        reference = project_root / reference
    if reference.is_file():
        rows.append(collect(reference, "resnet18 (checkpoint2)"))
    for summary_path in summaries:
        rows.append(collect(summary_path, summary_path.parent.name))

    write_csv(suite_root / "comparison.csv", rows)
    write_markdown(suite_root / "comparison.md", rows)
    figure_dir = project_root / "reports" / "figures" / args.name
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_plot(figure_dir / "architecture_comparison.png", rows)
    print(f"Compared {len(rows)} models")
    print(f"  table: {suite_root / 'comparison.csv'}")
    print(f"  report: {suite_root / 'comparison.md'}")
    print(f"  figure: {figure_dir / 'architecture_comparison.png'}")


if __name__ == "__main__":
    main()
