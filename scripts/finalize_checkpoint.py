#!/usr/bin/env python
"""Package one completed experiment into a numbered, Git-trackable milestone."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


CHECKPOINT_NAME = re.compile(r"checkpoint[1-9][0-9]*$")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_history_csv(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(history[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history)


def save_loss_plot(path: Path, history: list[dict], best_epoch: int, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axis.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation")
    axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
    axis.set(title=title, xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_accuracy_plot(
    path: Path, history: list[dict], best_epoch: int, title: str
) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        epochs, [row["train_accuracy"] for row in history], marker="o", label="Train"
    )
    axis.plot(
        epochs, [row["val_accuracy"] for row in history], marker="o", label="Validation"
    )
    axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
    axis.set(title=title, xlabel="Epoch", ylabel="Accuracy", ylim=(0.0, 1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Numbered label such as checkpoint1")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    if not CHECKPOINT_NAME.fullmatch(args.name):
        raise SystemExit("--name must be checkpoint1, checkpoint2, ...")

    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    required = {
        "history": run_dir / "history.json",
        "metadata": run_dir / "metadata.json",
        "validation": run_dir / "val_metrics.json",
        "checkpoint": run_dir / "best.pt",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise SystemExit(
            "Finalize only after training and validation. Missing: " + ", ".join(missing)
        )

    history = read_json(required["history"])
    metadata = read_json(required["metadata"])
    validation = read_json(required["validation"])
    if not history:
        raise SystemExit("Training history is empty.")
    expected = {"epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"}
    missing_fields = expected.difference(history[0])
    if missing_fields:
        raise SystemExit(f"History lacks graph fields: {sorted(missing_fields)}")

    best_row = max(history, key=lambda row: row["val_accuracy"])
    best_epoch = int(best_row["epoch"])
    result_dir = project_root / "results" / args.name
    figure_dir = project_root / "reports" / "figures" / args.name
    checkpoint_dir = project_root / "artifacts" / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    local_checkpoint = checkpoint_dir / f"{args.name}.pt"
    shutil.copy2(required["checkpoint"], local_checkpoint)
    save_history_csv(result_dir / "history.csv", history)
    shutil.copy2(required["validation"], result_dir / "validation_metrics.json")
    save_loss_plot(
        figure_dir / "loss_curve.png",
        history,
        best_epoch,
        f"{args.name}: Training and validation loss",
    )
    save_accuracy_plot(
        figure_dir / "accuracy_curve.png",
        history,
        best_epoch,
        f"{args.name}: Training and validation accuracy",
    )

    summary = {
        "checkpoint_name": args.name,
        "source_run": metadata["config"]["run_name"],
        "checkpoint_path": str(local_checkpoint.relative_to(project_root)),
        "checkpoint_sha256": file_sha256(local_checkpoint),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy_from_history": best_row["val_accuracy"],
        "best_validation_macro_f1_from_history": best_row.get("val_macro_f1"),
        "saved_checkpoint_validation": validation,
        "final_epoch": history[-1],
        "config": metadata["config"],
        "environment": metadata["environment"],
        "complexity": metadata["complexity"],
    }
    write_json(result_dir / "summary.json", summary)

    print(f"Finalized {args.name}")
    print(f"  summary: {result_dir / 'summary.json'}")
    print(f"  loss graph: {figure_dir / 'loss_curve.png'}")
    print(f"  accuracy graph: {figure_dir / 'accuracy_curve.png'}")
    print(f"  local checkpoint: {local_checkpoint}")


if __name__ == "__main__":
    main()
