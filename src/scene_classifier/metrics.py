from __future__ import annotations

import torch


def metrics_from_confusion(confusion: torch.Tensor, class_names: list[str]) -> dict:
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    recall = true_positive / support.clamp_min(1)
    precision = true_positive / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    total = support.sum().item()
    accuracy = true_positive.sum().item() / total if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": f1.mean().item(),
        "per_class_recall": {
            name: recall[index].item() for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }
