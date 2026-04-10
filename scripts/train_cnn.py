"""
Fine-tuning script for the EfficientNet-B3 mushroom CNN classifier.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR   = PROJECT_ROOT / "data" / "raw" / "images"
ARTIFACTS    = PROJECT_ROOT / "artifacts"
WEIGHTS_OUT  = ARTIFACTS / "cnn_weights.pt"
HISTORY_OUT  = ARTIFACTS / "cnn_training_history.json"

# Map species_id folder names → TARGET_SPECIES label
SPECIES_ID_TO_LABEL: Dict[str, str] = {
    "AM.MU": "Fly Agaric",
    "CA.CI": "Chanterelle",
    "HY.PS": "False Chanterelle",
    "BO.ED": "Porcini",
    "BO.BA": "Other Boletus",
    "AM.VI": "Amanita virosa",
    "CR.CO": "Black Trumpet",
}

LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(SPECIES_ID_TO_LABEL.values())}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


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


def build_datasets(samples: List[Tuple[Path, int]], val_fraction: float = 0.2):
    """Split samples into train/val and wrap in DataLoader objects."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_fraction))
    train_samples = samples[n_val:]
    val_samples   = samples[:n_val]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        batch_size=args.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        MushroomDataset(val_samples, val_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
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
    if len(samples) < len(SPECIES_ID_TO_LABEL):
        logger.error(
            "Only %d images found across all species — need at least 1 per species. "
            "Add images to data/raw/images/<species_id>/",
            len(samples),
        )
        return
    logger.info("Total images: %d", len(samples))

    train_loader, val_loader = build_datasets(samples)

    # ── Build model ───────────────────────────────────────────────────
    # EfficientNet-B3 pretrained on ImageNet-21k (includes fungi classes)
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=True,
        num_classes=len(LABEL_TO_IDX),
    )
    model.to(device)

    # ── Phase 1: train head only (freeze backbone) ────────────────────
    logger.info("Phase 1 — training classification head (backbone frozen)")
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name or "head" in name

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    patience_counter = 0

    for phase, total_epochs, unfreeze in [
        ("head-only",  min(10, args.epochs // 3),  False),
        ("fine-tune",  args.epochs,                 True),
    ]:
        if unfreeze:
            logger.info("Phase 2 — fine-tuning full network (lower LR)")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)

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
                t_loss    += loss.item() * imgs.size(0)
                t_correct += (out.argmax(1) == labels).sum().item()
                t_total   += imgs.size(0)

            # Validate
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out = model(imgs)
                    v_loss    += criterion(out, labels).item() * imgs.size(0)
                    v_correct += (out.argmax(1) == labels).sum().item()
                    v_total   += imgs.size(0)

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
                ARTIFACTS.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model_state_dict": model.state_dict(), "val_acc": v_acc},
                    WEIGHTS_OUT,
                )
                logger.info("  ✓ Saved best model (val_acc=%.3f)", v_acc)
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    logger.info("Early stopping (patience=5)")
                    break

    logger.info("Training complete. Best val_acc=%.3f", best_val_acc)
    logger.info("Weights saved to: %s", WEIGHTS_OUT)

    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("History saved to: %s", HISTORY_OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B3 for mushroom ID")
    parser.add_argument("--epochs",     type=int,   default=20,   help="Total fine-tune epochs")
    parser.add_argument("--batch-size", type=int,   default=8,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=3e-4, help="Initial learning rate")
    args = parser.parse_args()
    train(args)
