#!/usr/bin/env python
"""Compare transfer-learning scopes across the checkpoint-5 architectures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ARCHITECTURES = [
    "resnet18",
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
    "resnext50_32x4d",
]
ARCHITECTURE_LABELS = {
    "resnet18": "ResNet-18",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
    "resnext50_32x4d": "ResNeXt-50",
}
SETUPS = [
    "scratch_all",
    "pretrained_all",
    "pretrained_last_stage",
    "pretrained_head_only",
]
SETUP_LABELS = {
    "scratch_all": "Scratch/all",
    "pretrained_all": "Pretrained/all",
    "pretrained_last_stage": "Pretrained/last stage",
    "pretrained_head_only": "Pretrained/head only",
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


def identify_setup(summary: dict) -> str:
    model = summary["config"]["model"]
    if not bool(model["pretrained"]):
        return "scratch_all"
    scope = model["trainable_scope"]
    if scope == "all":
        return "pretrained_all"
    if scope in {"last_stage_and_head", "layer4_and_head"}:
        return "pretrained_last_stage"
    if scope == "head_only":
        return "pretrained_head_only"
    raise ValueError(f"Unknown trainable scope: {scope}")


def identify_architecture(summary: dict, reference: bool) -> str:
    if reference:
        return "resnext50_32x4d"
    return summary["config"]["model"]["name"].removesuffix("_transfer")


def collect(summary_path: Path, reference: bool = False) -> dict:
    summary = read_json(summary_path)
    config = summary["config"]
    validation = summary["saved_checkpoint_validation"]
    complexity = summary["complexity"]
    architecture = identify_architecture(summary, reference)
    setup = identify_setup(summary)
    return {
        "architecture": architecture,
        "architecture_label": ARCHITECTURE_LABELS[architecture],
        "setup": setup,
        "setup_label": SETUP_LABELS[setup],
        "source_checkpoint": "checkpoint4" if reference else "checkpoint5",
        "learning_rate": config["training"]["learning_rate"],
        "best_epoch": summary["best_epoch"],
        "epochs_completed": summary["epochs_completed"],
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


def value_lookup(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["architecture"], row["setup"]): row for row in rows}


def write_markdown(path: Path, rows: list[dict], includes_reference: bool) -> None:
    lookup = value_lookup(rows)
    available_architectures = [
        architecture
        for architecture in ARCHITECTURES
        if any(row["architecture"] == architecture for row in rows)
    ]
    lines = [
        "# Checkpoint 5 multi-architecture transfer matrix",
        "",
        "All checkpoint-5 runs use the same leakage-controlled split, 224x224 RGB pipeline, ImageNet normalization, augmentation, seed, batch size, regularization, scheduler, and maximum epoch budget. Learning rates are adapted to the optimization scope.",
        "",
    ]
    if includes_reference:
        lines.extend(
            [
                "The ResNeXt-50 rows are the directly comparable checkpoint-4 results and were not retrained.",
                "",
            ]
        )

    lines.extend(
        [
            "## Validation accuracy matrix",
            "",
            "| Architecture | Scratch/all | Pretrained/all | Pretrained/last stage | Pretrained/head only |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for architecture in available_architectures:
        values = []
        for setup in SETUPS:
            row = lookup.get((architecture, setup))
            values.append("--" if row is None else f"{100 * row['validation_accuracy']:.2f}%")
        lines.append(
            f"| {ARCHITECTURE_LABELS[architecture]} | " + " | ".join(values) + " |"
        )

    lines.extend(
        [
            "",
            "## Detailed results",
            "",
            "| Architecture | Setup | Val. accuracy | Macro F1 | Trainable / total | Peak GPU | Train time |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    ranked = sorted(rows, key=lambda row: row["validation_accuracy"], reverse=True)
    for row in ranked:
        lines.append(
            f"| {row['architecture_label']} | {row['setup_label']} | "
            f"{100 * row['validation_accuracy']:.2f}% | "
            f"{100 * row['validation_macro_f1']:.2f}% | "
            f"{row['trainable_parameters'] / 1e6:.3f}M / "
            f"{row['total_parameters'] / 1e6:.3f}M | "
            f"{row['peak_gpu_memory_mib']:.1f} MiB | "
            f"{row['training_seconds']:.1f}s |"
        )
    lines.extend(
        [
            "",
            "Freezing changes training cost, not inference MACs: every setup still executes its complete backbone at inference.",
            "",
            "The labeled test set is intentionally excluded from model selection.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_plot(path: Path, rows: list[dict]) -> None:
    lookup = value_lookup(rows)
    architectures = [
        architecture
        for architecture in ARCHITECTURES
        if any(row["architecture"] == architecture for row in rows)
    ]
    x = np.arange(len(architectures))
    width = 0.19
    figure, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    for index, setup in enumerate(SETUPS):
        offset = (index - 1.5) * width
        accuracy = [
            lookup.get((architecture, setup), {}).get("validation_accuracy", np.nan)
            for architecture in architectures
        ]
        memory = [
            lookup.get((architecture, setup), {}).get("peak_gpu_memory_mib", np.nan)
            for architecture in architectures
        ]
        axes[0].bar(x + offset, accuracy, width, label=SETUP_LABELS[setup])
        axes[1].bar(x + offset, memory, width, label=SETUP_LABELS[setup])

    axes[0].set(
        ylabel="Validation accuracy",
        title="Architecture and transfer-learning strategy",
        ylim=(0, 1),
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].set(ylabel="Peak training GPU memory (MiB)")
    axes[1].set_xticks(x, [ARCHITECTURE_LABELS[item] for item in architectures])
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="checkpoint5")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--without-resnext-reference",
        action="store_true",
        help="Exclude the directly comparable checkpoint-4 ResNeXt runs.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    suite_root = project_root / "results" / args.name
    summaries = sorted(suite_root.glob("*/summary.json"))
    if not summaries:
        raise SystemExit(f"No finalized variants found beneath {suite_root}")
    rows = [collect(summary_path) for summary_path in summaries]

    includes_reference = False
    reference_root = project_root / "results" / "checkpoint4"
    if not args.without_resnext_reference and reference_root.is_dir():
        for summary_path in sorted(reference_root.glob("*/summary.json")):
            rows.append(collect(summary_path, reference=True))
            includes_reference = True

    architecture_order = {name: index for index, name in enumerate(ARCHITECTURES)}
    setup_order = {name: index for index, name in enumerate(SETUPS)}
    rows.sort(
        key=lambda row: (
            architecture_order[row["architecture"]], setup_order[row["setup"]]
        )
    )
    suite_root.mkdir(parents=True, exist_ok=True)
    write_csv(suite_root / "comparison.csv", rows)
    write_markdown(suite_root / "comparison.md", rows, includes_reference)
    figure_dir = project_root / "reports" / "figures" / args.name
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_plot(figure_dir / "transfer_matrix.png", rows)
    print(f"Compared {len(rows)} architecture/setup combinations")
    print(f"  table: {suite_root / 'comparison.csv'}")
    print(f"  report: {suite_root / 'comparison.md'}")
    print(f"  figure: {figure_dir / 'transfer_matrix.png'}")


if __name__ == "__main__":
    main()
