#!/usr/bin/env python3
"""
Visualize every step of the pipeline for test images:
  1. Original above/below photos
  2. Raw YOLO segmentation masks (overlay + individual binary masks)
  3. Processed part masks (after build_part_masks quality gates)
  4. Overlay of accepted masks on originals
  5. Case + extracted traits text summary

Saves one figure per specimen to data/testing/output/
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
sys.path.insert(0, str(PROJECT_ROOT))

from models.mushroom_segmenter import get_segmenter, detect_case, detect_case_from_masks
from models.visual_trait_extractor import extract
from models.yolo_part_masks import build_part_masks, _mask_quality
from config import trait_config as tc


OUTPUT_DIR = PROJECT_ROOT / "data" / "testing" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PART_COLORS = {
    "cap": (255, 0, 0),        # red
    "stem": (0, 255, 0),       # green
    "underside": (0, 0, 255),  # blue
    "coral": (255, 0, 255),    # magenta
    "whole": (255, 255, 0),    # yellow
}

RAW_CLASS_COLORS = {
    "Cap": (255, 0, 0),
    "Stem": (0, 255, 0),
    "Underside": (0, 0, 255),
    "Coral": (255, 0, 255),
}


def load_bgr(path: Path) -> np.ndarray:
    pil = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def overlay_mask_on_image(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.4) -> np.ndarray:
    overlay = img.copy()
    bool_mask = mask > 0
    overlay[bool_mask] = (
        overlay[bool_mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def draw_raw_instances_overlay(ax, img_bgr: np.ndarray, instances: List[Dict[str, Any]], title: str):
    canvas = img_bgr.copy()
    for inst in instances:
        cls = inst.get("class_name", "unknown")
        mask = inst.get("mask")
        if mask is None or cls not in RAW_CLASS_COLORS:
            continue
        color = RAW_CLASS_COLORS[cls]
        canvas = overlay_mask_on_image(canvas, mask, color, alpha=0.35)

    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    labels = []
    for inst in instances:
        cls = inst.get("class_name", "unknown")
        conf = inst.get("model_confidence", 0.0)
        if cls not in RAW_CLASS_COLORS:
            continue
        labels.append(f"{cls} ({conf:.2f})")
    if labels:
        ax.text(0.02, 0.98, "\n".join(labels), transform=ax.transAxes,
                fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))


def draw_raw_binary_masks(ax, img_bgr: np.ndarray, instances: List[Dict[str, Any]], title_prefix: str):
    """Draw a grid of raw binary masks with quality metrics."""
    ax.set_title(f"{title_prefix} raw binary masks", fontsize=9)
    ax.axis("off")

    valid = [i for i in instances if i.get("mask") is not None and i.get("class_name") in RAW_CLASS_COLORS]
    n = len(valid)
    if n == 0:
        ax.text(0.5, 0.5, "No raw masks", ha="center", va="center", fontsize=10)
        return

    # Sub-grid inside this axis
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    sub_gs = ax.get_subplotspec().subgridspec(rows, cols, wspace=0.1, hspace=0.3)

    for idx, inst in enumerate(valid):
        sub_ax = ax.figure.add_subplot(sub_gs[idx // cols, idx % cols])
        mask = inst["mask"]
        cls = inst.get("class_name", "?")
        conf = inst.get("model_confidence", 0.0)
        q = _mask_quality(mask, conf)

        # Show mask as grayscale on dark background
        display = np.zeros((*mask.shape, 3), dtype=np.uint8)
        display[mask > 0] = [200, 200, 200]

        sub_ax.imshow(display)
        sub_ax.set_title(
            f"{cls}\nc={conf:.2f}, ar={q['area_ratio']:.3f}\nfrag={q['fragmentation']}, hole={q['hole_ratio']:.2f}",
            fontsize=6,
        )
        sub_ax.axis("off")


def draw_processed_masks_grid(ax_list, img_bgr, part_masks, prefix):
    part_keys = ["cap", "stem", "underside", "coral", "whole"]
    for ax, key in zip(ax_list, part_keys):
        info = part_masks.get(key)
        if info is None:
            ax.set_facecolor("lightgray")
            ax.text(0.5, 0.5, f"{prefix}\n{key}\n(rejected)", ha="center", va="center", fontsize=8)
            ax.axis("off")
            continue

        mask = info["mask"]
        color = PART_COLORS.get(key, (128, 128, 128))
        overlay = overlay_mask_on_image(img_bgr, mask, color, alpha=0.5)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        title = f"{prefix} {key}\nconf={info['confidence']}, inst={info['instance_count']}"
        if info.get('rejected_count', 0) > 0:
            title += f", rej={info['rejected_count']}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")


def draw_trait_text(ax, traits, case_info):
    lines = []
    lines.append(f"Case: {case_info.get('case', 'unknown')} (conf={case_info.get('confidence', 0)})")
    lines.append(f"Detected parts: {case_info.get('detected_parts', [])}")
    lines.append("")
    lines.append("Traits:")
    for k in ["dominant_color", "cap_color", "cap_shape", "cap_surface",
              "stem_color", "stem_surface", "stem_ring",
              "underside_color", "hymenophore_type",
              "whole_color", "coral_branching", "puffball_surface",
              "surface_texture", "brightness"]:
        v = traits.get(k)
        if v is not None and v != "unknown":
            lines.append(f"  {k}: {v}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def process_specimen(name, above_path, below_path, use_part_aware=False):
    print(f"Processing {name} ...")

    seg = get_segmenter()
    above_bgr = load_bgr(above_path)
    below_bgr = load_bgr(below_path) if below_path.exists() else above_bgr

    above_bytes = above_path.read_bytes()
    below_bytes = below_path.read_bytes() if below_path.exists() else above_bytes

    above_seg = seg.segment(above_bytes)
    below_seg = seg.segment(below_bytes)

    H_a, W_a = above_bgr.shape[:2]
    H_b, W_b = below_bgr.shape[:2]

    above_masks = build_part_masks(above_seg.get("instances", []), (H_a, W_a))
    below_masks = build_part_masks(below_seg.get("instances", []), (H_b, W_b))

    if use_part_aware:
        case = detect_case_from_masks(above_masks, below_masks)
        traits = extract(above_bytes, part_masks=above_masks)["visible_traits"]
    else:
        case = detect_case(above_seg.get("instances", []), below_seg.get("instances", []))
        traits = extract(above_bytes)["visible_traits"]

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 6, hspace=0.35, wspace=0.2)

    # Row 0: Originals + Raw overlays
    ax0a = fig.add_subplot(gs[0, 0])
    ax0a.imshow(cv2.cvtColor(above_bgr, cv2.COLOR_BGR2RGB))
    ax0a.set_title(f"{name} — Above (original)", fontsize=10)
    ax0a.axis("off")

    ax0b = fig.add_subplot(gs[0, 1])
    ax0b.imshow(cv2.cvtColor(below_bgr, cv2.COLOR_BGR2RGB))
    ax0b.set_title(f"{name} — Below (original)", fontsize=10)
    ax0b.axis("off")

    ax0c = fig.add_subplot(gs[0, 2:4])
    draw_raw_instances_overlay(ax0c, above_bgr, above_seg.get("instances", []), "Raw YOLO — Above")

    ax0d = fig.add_subplot(gs[0, 4:6])
    draw_raw_instances_overlay(ax0d, below_bgr, below_seg.get("instances", []), "Raw YOLO — Below")

    # Row 1: Raw binary masks
    ax1a = fig.add_subplot(gs[1, 0:3])
    draw_raw_binary_masks(ax1a, above_bgr, above_seg.get("instances", []), "Above")

    ax1b = fig.add_subplot(gs[1, 3:5])
    draw_raw_binary_masks(ax1b, below_bgr, below_seg.get("instances", []), "Below")

    ax1c = fig.add_subplot(gs[1, 5])
    draw_trait_text(ax1c, traits, case)

    # Row 2: Processed above masks
    ax2a = fig.add_subplot(gs[2, 0])
    ax2b = fig.add_subplot(gs[2, 1])
    ax2c = fig.add_subplot(gs[2, 2])
    ax2d = fig.add_subplot(gs[2, 3])
    ax2e = fig.add_subplot(gs[2, 4])
    draw_processed_masks_grid([ax2a, ax2b, ax2c, ax2d, ax2e], above_bgr, above_masks, "Above")

    ax2f = fig.add_subplot(gs[2, 5])
    ax2f.axis("off")

    # Row 3: Processed below masks
    ax3a = fig.add_subplot(gs[3, 0])
    ax3b = fig.add_subplot(gs[3, 1])
    ax3c = fig.add_subplot(gs[3, 2])
    ax3d = fig.add_subplot(gs[3, 3])
    ax3e = fig.add_subplot(gs[3, 4])
    draw_processed_masks_grid([ax3a, ax3b, ax3c, ax3d, ax3e], below_bgr, below_masks, "Below")

    ax3f = fig.add_subplot(gs[3, 5])
    ax3f.axis("off")

    suffix = "_partaware" if use_part_aware else "_legacy"
    out_path = OUTPUT_DIR / f"{name}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    test_dir = PROJECT_ROOT / "data" / "testing"
    images = sorted(test_dir.glob("*_above.jpg"))

    if not images:
        print("No *_above.jpg images found in data/testing/")
        return

    for above_path in images:
        name = above_path.stem.replace("_above", "")
        below_path = test_dir / f"{name}_below.jpg"
        process_specimen(name, above_path, below_path, use_part_aware=False)
        process_specimen(name, above_path, below_path, use_part_aware=True)

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
