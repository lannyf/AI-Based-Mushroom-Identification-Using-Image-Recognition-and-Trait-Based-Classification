#!/usr/bin/env python3
"""
Train YOLOv8n-seg on mushroom segmentation dataset.

Usage:
    # On GPU (Google Colab or local CUDA)
    python scripts/train_yolov8_seg.py --dataset data/segmentation/dataset.yaml --device 0

    # On CPU (very slow, 10-24 hours expected)
    python scripts/train_yolov8_seg.py --dataset data/segmentation/dataset.yaml --device cpu

After training:
    - Best checkpoint copied to: artifacts/yolov8_seg_ft.pt
    - Metadata written to:      artifacts/yolov8_seg_ft_metadata.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def train(
    dataset_yaml: str,
    device: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 20,
    close_mosaic: int = 10,
    lr0: float = 0.001,
    project: str = "artifacts/yolov8_seg_runs",
    name: str = "mushroom_seg",
    weights: str = "yolov8n-seg.pt",
    exist_ok: bool = True,
) -> Optional[Any]:
    """Run YOLOv8-seg training and return the best model path."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Error: ultralytics not installed. {exc}", file=sys.stderr)
        return None

    print(f"Loading base model: {weights}")
    model = YOLO(weights)

    print(f"Starting training on {dataset_yaml}")
    print(f"  epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}")
    print(f"  patience={patience}, close_mosaic={close_mosaic}, lr0={lr0}")
    print()

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        close_mosaic=close_mosaic,
        lr0=lr0,
        project=project,
        name=name,
        exist_ok=exist_ok,
        device=device,
        # Augmentation defaults are reasonable; we keep YOLO's built-in HSV
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    return results


def copy_best_checkpoint(
    project: str,
    name: str,
    dest_path: str,
) -> Optional[Path]:
    """Copy best.pt from the training run to the destination."""
    run_dir = Path(project) / name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"Warning: best.pt not found at {best_pt}", file=sys.stderr)
        # Try last.pt as fallback
        last_pt = run_dir / "weights" / "last.pt"
        if last_pt.exists():
            best_pt = last_pt
        else:
            return None

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, dest)
    print(f"Copied best checkpoint to: {dest}")
    return dest


def write_metadata(
    dest_path: str,
    dataset_yaml: str,
    config: Dict[str, Any],
) -> None:
    """Write training metadata JSON next to the checkpoint."""
    dataset_yaml_path = Path(dataset_yaml)
    meta_path = Path(dest_path).with_suffix("_metadata.json")

    # Count train/val images from the dataset directory
    train_dir = dataset_yaml_path.parent / "images" / "train"
    val_dir = dataset_yaml_path.parent / "images" / "val"
    train_images = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
    val_images = len(list(val_dir.glob("*"))) if val_dir.exists() else 0

    metadata: Dict[str, Any] = {
        "model": "yolov8n-seg",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "roboflow-4class",
        "dataset_yaml": str(dataset_yaml_path.resolve()),
        "train_images": train_images,
        "val_images": val_images,
        "epochs": config.get("epochs", 100),
        "imgsz": config.get("imgsz", 640),
        "batch": config.get("batch", 8),
        "patience": config.get("patience", 20),
        "close_mosaic": config.get("close_mosaic", 10),
        "lr0": config.get("lr0", 0.001),
        "device": config.get("device", "cpu"),
        "class_map": {"0": "mushroom"},
        "source_checkpoint": config.get("weights", "yolov8n-seg.pt"),
        "project": config.get("project", "artifacts/yolov8_seg_runs"),
        "name": config.get("name", "mushroom_seg"),
    }

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata written to: {meta_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8n-seg on mushroom segmentation dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset.yaml")
    parser.add_argument("--device", default="cpu", help="Device: 'cpu', '0', '0,1,2,3', etc.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--project", default="artifacts/yolov8_seg_runs")
    parser.add_argument("--name", default="mushroom_seg")
    parser.add_argument("--weights", default="yolov8n-seg.pt")
    parser.add_argument("--dest", default="artifacts/yolov8_seg_ft.pt", help="Destination path for best checkpoint")
    parser.add_argument("--exist-ok", action="store_true", default=True, help="Overwrite existing training run")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Error: dataset.yaml not found: {args.dataset}", file=sys.stderr)
        return 1

    config = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "close_mosaic": args.close_mosaic,
        "lr0": args.lr0,
        "project": args.project,
        "name": args.name,
        "weights": args.weights,
        "device": args.device,
    }

    results = train(
        dataset_yaml=args.dataset,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        weights=args.weights,
        exist_ok=args.exist_ok,
    )

    if results is None:
        print("Training failed or was interrupted.", file=sys.stderr)
        return 1

    best_path = copy_best_checkpoint(args.project, args.name, args.dest)
    if best_path is None:
        print("Warning: could not copy best checkpoint.", file=sys.stderr)
        return 1

    write_metadata(args.dest, args.dataset, config)
    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
