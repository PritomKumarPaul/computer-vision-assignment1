from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .utils import seed_worker


class ManifestDataset(Dataset):
    """Image dataset backed by a checked-in CSV split manifest."""

    def __init__(self, image_root: Path, manifest: Path, transform: Callable):
        self.image_root = image_root
        self.transform = transform
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            self.samples = list(csv.DictReader(handle))
        if not self.samples:
            raise ValueError(f"Manifest is empty: {manifest}")

        labels = sorted({int(row["label"]) for row in self.samples})
        if labels != list(range(len(labels))):
            raise ValueError(f"Labels must be contiguous from zero; got {labels}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        path = self.image_root / row["relative_path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label"])


def build_transforms(data_config: dict):
    profile = data_config["transform"]
    size = int(data_config["image_size"])
    color_mode = data_config.get("color_mode", "grayscale")
    if profile == "starter" and color_mode == "grayscale":
        shared = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        return shared, shared

    if profile == "resnet_regularized" and color_mode == "rgb":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=(0.75, 1.0),
                    ratio=(0.80, 1.25),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                transforms.RandomErasing(
                    p=0.20, scale=(0.02, 0.12), ratio=(0.5, 2.0), value="random"
                ),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(size + 16, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        return train_transform, eval_transform

    raise ValueError(
        f"Unsupported transform/color combination: profile={profile!r}, "
        f"color_mode={color_mode!r}"
    )


def class_names_from_manifest(manifest: Path) -> list[str]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {int(row["label"]): row["class_name"] for row in rows}
    return [mapping[index] for index in range(len(mapping))]


def build_loaders(config: dict, project_root: Path):
    data_config = config["data"]
    data_root = project_root / data_config["root"]
    train_manifest = project_root / data_config["train_manifest"]
    val_manifest = project_root / data_config["val_manifest"]
    train_transform, eval_transform = build_transforms(data_config)

    train_dataset = ManifestDataset(data_root / "train", train_manifest, train_transform)
    val_dataset = ManifestDataset(data_root / "train", val_manifest, eval_transform)
    class_names = class_names_from_manifest(train_manifest)

    generator = torch.Generator().manual_seed(int(config["seed"]))
    common = {
        "batch_size": int(data_config["batch_size"]),
        "num_workers": int(data_config.get("num_workers", 2)),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, class_names


def build_test_loader(config: dict, project_root: Path, class_names: list[str]):
    data_config = config["data"]
    _, eval_transform = build_transforms(data_config)
    test_root = project_root / data_config["root"] / "test"
    dataset = datasets.ImageFolder(test_root, transform=eval_transform)
    if dataset.classes != class_names:
        raise ValueError(
            f"Test classes do not match checkpoint classes: {dataset.classes} != {class_names}"
        )
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    return loader
