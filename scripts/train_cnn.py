"""
Fine-tuning script for the mushroom CNN classifier.

Expected dataset layout:
    data/raw/images/
        AM.MU/          ← Amanita muscaria  (Fly Agaric)
            img001.jpg
            img002.jpg
            ...
        CA.CI/          ← Cantharellus cibarius  (Chanterelle)
            ...
        HY.PS/          ← False Chanterelle
        BO.ED/          ← Porcini
        BO.BA/          ← Other Boletus (use any Boletus species_id)
        AM.VI/          ← Amanita virosa
        CR.CO/          ← Black Trumpet

At least 10 images per species are needed; 20–40 gives solid fine-tuning.

Usage
-----
    python scripts/train_cnn.py
    python scripts/train_cnn.py --epochs 30 --batch-size 8 --lr 3e-4

The trained weights are saved to artifacts/cnn_weights.pt and the API
will pick them up automatically on next restart.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

from config.image_model_config import (
    ARTIFACTS_DIR,
    BASE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    FINETUNE_LR_FACTOR,
    HEAD_ONLY_EPOCHS_FRACTION,
    HEAD_ONLY_LR_FACTOR,
    HISTORY_PATH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    NUM_CLASSES,
    RESIZE_SIZE,
    SPECIES,
    TRAIN_AUGMENTATION,
    VAL_FRACTION,
    WEIGHTS_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"

# Map species_id folder names → class labels.
# All values must exist in config.SPECIES.
SPECIES_ID_TO_LABEL: Dict[str, str] = {
    "AM.MU": "Fly Agaric",
    "CA.CI": "Chanterelle",
    "HY.PS": "False Chanterelle",
    "BO.ED": "Porcini",
    "BO.BA": "Other Boletus",
    "AM.VI": "Amanita virosa",
    "CR.CO": "Black Trumpet",
}

# Validate mapping against canonical species list from config
for _sid, _lbl in SPECIES_ID_TO_LABEL.items():
    if _lbl not in SPECIES:
        raise ValueError(
            f"SPECIES_ID_TO_LABEL maps {_sid} to '{_lbl}' which is not in config.SPECIES"
        )

LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(SPECIES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_samples(images_dir: Path) -> List[Tuple[Path, int]]:
    """Scan the images directory and return (path, class_idx) pairs."""
    samples: List[Tuple[Path, int]] = []
    for species_id, label in SPECIES_ID_TO_LABEL.items():
        folder = images_dir / species_id
        if not folder.is_dir():
            logger.warning("Missing folder for %s (%s) — skipping", label, species_id)
            continue
        imgs = [p for p in folder.iterdir() if p.suffix in IMAGE_EXTS]
        logger.info("  %s: %d images", label, len(imgs))
        samples.extend((p, LABEL_TO_IDX[label]) for p in imgs)
    return samples


def build_datasets(samples: List[Tuple[Path, int]], batch_size: int, val_fraction: float = VAL_FRACTION):
    """Split samples into train/val and wrap in DataLoader objects."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_fraction))
    train_samples = samples[n_val:]
    val_samples = samples[:n_val]

    crop_size = INPUT_SIZE[0]  # assumes square input (height == width)
    resize_size = RESIZE_SIZE
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    aug = TRAIN_AUGMENTATION

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=aug["random_resized_crop_scale"]),
        transforms.RandomHorizontalFlip() if aug["random_horizontal_flip"] else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(*aug["color_jitter"]),
        transforms.RandomRotation(aug["random_rotation"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    class MushroomDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = Image.open(path).convert("RGB")
            return self.transform(img), label

    train_loader = DataLoader(
        MushroomDataset(train_samples, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        MushroomDataset(val_samples, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )
    return train_loader, val_loader


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Collect data ──────────────────────────────────────────────────
    logger.info("Scanning %s ...", IMAGES_DIR)
    samples = collect_samples(IMAGES_DIR)
    if len(samples) < NUM_CLASSES:
        logger.error(
            "Only %d images found across all species — need at least 1 per species. "
            "Add images to data/raw/images/<species_id>/",
            len(samples),
        )
        return
    logger.info("Total images: %d", len(samples))

    train_loader, val_loader = build_datasets(samples, batch_size=args.batch_size)

    # ── Build model ───────────────────────────────────────────────────
    logger.info("Building %s (num_classes=%d) ...", BASE_MODEL, NUM_CLASSES)
    model = timm.create_model(
        BASE_MODEL,
        pretrained=True,
        num_classes=NUM_CLASSES,
    )
    model.to(device)

    # ── Phase 1: train head only (freeze backbone) ────────────────────
    logger.info("Phase 1 — training classification head (backbone frozen)")
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name or "head" in name

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * HEAD_ONLY_LR_FACTOR,
    )
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    patience_counter = 0

    head_only_epochs = max(1, int(args.epochs * HEAD_ONLY_EPOCHS_FRACTION))

    for phase, total_epochs, unfreeze in [
        ("head-only", head_only_epochs, False),
        ("fine-tune", args.epochs, True),
    ]:
        if unfreeze:
            logger.info("Phase 2 — fine-tuning full network (lower LR)")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr * FINETUNE_LR_FACTOR
            )

        for epoch in range(total_epochs):
            # Train
            model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * imgs.size(0)
                t_correct += (out.argmax(1) == labels).sum().item()
                t_total += imgs.size(0)

            # Validate
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs)
                    v_loss += criterion(out, labels).item() * imgs.size(0)
                    v_correct += (out.argmax(1) == labels).sum().item()
                    v_total += imgs.size(0)

            t_acc = t_correct / t_total
            v_acc = v_correct / v_total
            t_loss /= t_total
            v_loss /= v_total

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)

            logger.info(
                "[%s] Epoch %d/%d  train_loss=%.4f  train_acc=%.3f  "
                "val_loss=%.4f  val_acc=%.3f",
                phase, epoch + 1, total_epochs, t_loss, t_acc, v_loss, v_acc,
            )

            # Save best weights
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                patience_counter = 0
                ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model_state_dict": model.state_dict(), "val_acc": v_acc},
                    WEIGHTS_PATH,
                )
                logger.info("  ✓ Saved best model (val_acc=%.3f)", v_acc)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping (patience=%d)", EARLY_STOPPING_PATIENCE)
                    break

    logger.info("Training complete. Best val_acc=%.3f", best_val_acc)
    logger.info("Weights saved to: %s", WEIGHTS_PATH)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("History saved to: %s", HISTORY_PATH)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune mushroom CNN classifier")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Total fine-tune epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Initial learning rate")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
