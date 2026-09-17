from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .metrics import metrics_from_confusion
from .utils import write_json


def _synchronize(device: torch.device) -> None:
    """Make CUDA wall-clock measurements include queued GPU work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def _make_grad_scaler(enabled: bool):
    """Support both current and older PyTorch AMP scaler APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    class_names,
    criterion=None,
    progress_desc: str = "Validation",
) -> dict:
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(
        (len(class_names), len(class_names)), dtype=torch.int64
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()

    progress = tqdm(
        loader, desc=progress_desc, unit="batch", dynamic_ncols=True, leave=False
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        flat = labels.cpu() * len(class_names) + predictions.cpu()
        confusion += torch.bincount(
            flat, minlength=len(class_names) ** 2
        ).reshape(len(class_names), len(class_names))
        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            accuracy=f"{correct / total:.4f}",
        )

    _synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = metrics_from_confusion(confusion, class_names)
    metrics["loss"] = running_loss / total
    metrics["elapsed_seconds"] = elapsed
    metrics["images_per_second"] = total / elapsed
    metrics["peak_gpu_memory_mib"] = _peak_memory_mib(device)
    return metrics


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    use_amp: bool,
    progress_desc: str,
):
    model.train()
    for module_name in getattr(model, "_frozen_module_names", ()):
        model.get_submodule(module_name).eval()
    running_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = use_amp and device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()

    progress = tqdm(
        loader, desc=progress_desc, unit="batch", dynamic_ncols=True, leave=False
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            accuracy=f"{correct / total:.4f}",
        )

    _synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "elapsed_seconds": elapsed,
        "images_per_second": total / elapsed,
        "peak_gpu_memory_mib": _peak_memory_mib(device),
    }


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    config,
    class_names,
    device,
    run_dir: Path,
):
    training = config["training"]
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training.get("label_smoothing", 0.0))
    )
    epochs = int(training["epochs"])
    patience = training.get("early_stopping_patience")
    amp_enabled = bool(training.get("amp", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)

    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    _synchronize(device)
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp=bool(training.get("amp", False)),
            progress_desc=f"Train {epoch:02d}/{epochs}",
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            class_names,
            criterion,
            progress_desc=f"Valid {epoch:02d}/{epochs}",
        )
        if scheduler is not None:
            scheduler.step()

        _synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        total_seconds = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_seconds": train_metrics["elapsed_seconds"],
            "validation_seconds": val_metrics["elapsed_seconds"],
            "epoch_seconds": epoch_seconds,
            "total_seconds": total_seconds,
            "train_images_per_second": train_metrics["images_per_second"],
            "validation_images_per_second": val_metrics["images_per_second"],
            "train_peak_gpu_memory_mib": train_metrics["peak_gpu_memory_mib"],
            "validation_peak_gpu_memory_mib": val_metrics["peak_gpu_memory_mib"],
        }
        history.append(row)
        memory_text = ""
        if row["train_peak_gpu_memory_mib"] is not None:
            memory_text = f" | peak GPU {row['train_peak_gpu_memory_mib']:.1f} MiB"
        print(
            f"Epoch {epoch:02d}/{epochs} | train loss {row['train_loss']:.4f} "
            f"acc {row['train_accuracy']:.4f} | val loss {row['val_loss']:.4f} "
            f"acc {row['val_accuracy']:.4f} macro-F1 {row['val_macro_f1']:.4f} "
            f"| {epoch_seconds:.1f}s | train {row['train_images_per_second']:.1f} img/s "
            f"| val {row['validation_images_per_second']:.1f} img/s{memory_text}"
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            checkpoint = {
                "model_state": best_state,
                "config": config,
                "class_names": class_names,
                "best_epoch": best_epoch,
                "best_val_metrics": val_metrics,
            }
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        write_json(run_dir / "history.json", history)
        if patience is not None and epochs_without_improvement >= int(patience):
            print(f"Early stopping after {epoch} epochs.")
            break

    _synchronize(device)
    total_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_accuracy, total_seconds
