from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn

from .metrics import metrics_from_confusion
from .models import set_backbone_trainable
from .utils import write_json


@torch.inference_mode()
def evaluate(model, loader, device, class_names, criterion=None) -> dict:
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    confusion = torch.zeros(
        (len(class_names), len(class_names)), dtype=torch.int64
    )

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        flat = labels.cpu() * len(class_names) + predictions.cpu()
        confusion += torch.bincount(
            flat, minlength=len(class_names) ** 2
        ).reshape(len(class_names), len(class_names))

    metrics = metrics_from_confusion(confusion, class_names)
    metrics["loss"] = running_loss / total
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp: bool):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    for images, labels in loader:
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

    return {"loss": running_loss / total, "accuracy": correct / total}


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
    freeze_epochs = int(training.get("freeze_backbone_epochs", 0))
    model_name = config["model"]["name"]

    if freeze_epochs:
        set_backbone_trainable(model, model_name, trainable=False)

    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, model_name, trainable=True)
            print("Unfroze pretrained backbone.")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_amp=bool(training.get("amp", False)),
        )
        val_metrics = evaluate(model, val_loader, device, class_names, criterion)
        if scheduler is not None:
            scheduler.step()

        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | train loss {row['train_loss']:.4f} "
            f"acc {row['train_accuracy']:.4f} | val loss {row['val_loss']:.4f} "
            f"acc {row['val_accuracy']:.4f} macro-F1 {row['val_macro_f1']:.4f} "
            f"| {elapsed:.1f}s"
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

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_accuracy, time.perf_counter() - started
