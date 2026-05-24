"""
YOLO Part Mask Builder

Converts raw YOLOv8 segmentation instances into normalized, quality-gated
part masks: cap, stem, underside, and a whole-body union.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config import trait_config as tc

logger = logging.getLogger(__name__)

RAW_TO_PART_KEY = {
    "Cap": "cap",
    "Stem": "stem",
    "Underside": "underside",
}


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two bboxes (x, y, w, h)."""
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = aw * ah
    b_area = bw * bh
    union_area = a_area + b_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _bbox_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Pixel distance between bbox centroids."""
    acx = a[0] + a[2] / 2.0
    acy = a[1] + a[3] / 2.0
    bcx = b[0] + b[2] / 2.0
    bcy = b[1] + b[3] / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _mask_quality(mask: np.ndarray, confidence: float) -> Dict[str, Any]:
    H, W = mask.shape[:2]
    area = float(np.count_nonzero(mask > 0))
    area_ratio = area / float(H * W)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    frag = len(contours) if contours else 0

    hole_ratio = 0.0
    boundary_irregularity = 0.0
    if contours:
        perimeters = [cv2.arcLength(c, True) for c in contours]
        perim = sum(perimeters)
        areas = [cv2.contourArea(c) for c in contours]
        area_sum = sum(areas)
        boundary_irregularity = perim / area_sum if area_sum > 0 else 0.0
        if hierarchy is not None:
            hier = hierarchy[0]
            inner = sum(1 for h in hier if h[3] != -1)
            hole_ratio = float(inner) / float(len(hier)) if len(hier) > 0 else 0.0

    # Border contact
    border_contact = bool(
        np.any(mask[0, :] > 0)
        or np.any(mask[-1, :] > 0)
        or np.any(mask[:, 0] > 0)
        or np.any(mask[:, -1] > 0)
    )

    return {
        "area_ratio": area_ratio,
        "fragmentation": frag,
        "hole_ratio": hole_ratio,
        "boundary_irregularity": boundary_irregularity,
        "touches_border": border_contact,
        "yolo_confidence": confidence,
    }


def _passes_quality_gate(q: Dict[str, Any], part_key: str = "") -> bool:
    # Part-specific overrides for classical mushroom parts.
    min_area = getattr(tc, "STEM_MIN_AREA_RATIO", tc.MIN_AREA_RATIO) if part_key == "stem" else tc.MIN_AREA_RATIO
    if part_key == "underside":
        min_area = getattr(tc, "UNDERSIDE_MIN_AREA_RATIO", tc.MIN_AREA_RATIO)

    max_hole = getattr(tc, "CAP_MAX_HOLE_RATIO", tc.MAX_HOLE_RATIO) if part_key == "cap" else tc.MAX_HOLE_RATIO

    return (
        q["area_ratio"] >= min_area
        and q["fragmentation"] <= tc.MAX_FRAGMENTATION
        and q["hole_ratio"] <= max_hole
        and q["yolo_confidence"] >= tc.MIN_CONFIDENCE
    )


def build_part_masks(
    instances: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
) -> Dict[str, Dict[str, Any]]:
    """
    Convert YOLO instances into normalized part masks.

    Returns a dict with keys: cap, stem, underside, whole.
    Each value contains: mask, confidence, bbox, class_name, quality,
    instance_count, rejected_count, accepted_clusters.
    If a part is absent or fails quality gates, the entry is omitted.

    Cluster handling:
      - cap / stem / underside: tight-cluster merge (IoU > 0.05 or centroid
        distance < 50 px) to avoid multi-mushroom noise.
    """
    H, W = image_shape

    # Group by normalized part key
    groups: Dict[str, List[Dict[str, Any]]] = {
        "cap": [],
        "stem": [],
        "underside": [],
    }

    for inst in instances:
        raw_name = inst.get("class_name", "unknown")
        part_key = RAW_TO_PART_KEY.get(raw_name)
        if part_key is None:
            continue

        mask = inst.get("mask")
        if mask is None:
            mask = inst.get("cleaned_mask")
        if mask is None:
            continue

        # Normalize mask
        mask = (mask > 0).astype(np.uint8) * 255
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        bbox = inst.get("bbox") or _bbox_from_mask(mask)
        conf = float(inst.get("model_confidence", 0.0))

        groups[part_key].append({
            "mask": mask,
            "bbox": bbox,
            "confidence": conf,
            "class_name": raw_name,
        })

    result: Dict[str, Dict[str, Any]] = {}
    accepted_masks: List[np.ndarray] = []

    for part_key, items in groups.items():
        if not items:
            continue

        # Sort by confidence descending
        items.sort(key=lambda x: x["confidence"], reverse=True)

        rejected_count = 0
        accepted_clusters: List[Dict[str, Any]] = []

        # Cap / stem / underside: keep tight-cluster merge behavior
        merged_mask = items[0]["mask"].copy()
        merged_bbox = list(items[0]["bbox"])
        merged_conf = items[0]["confidence"]
        instance_count = 1

        for item in items[1:]:
            iou = _bbox_iou(tuple(merged_bbox), item["bbox"])
            dist = _bbox_distance(tuple(merged_bbox), item["bbox"])
            if iou > 0.05 or dist < 50:
                merged_mask = np.logical_or(merged_mask > 0, item["mask"] > 0).astype(np.uint8) * 255
                # Update enclosing bbox
                mx1, my1, mx2, my2 = _bbox_to_corners(tuple(merged_bbox))
                ix1, iy1, ix2, iy2 = _bbox_to_corners(item["bbox"])
                nx1, ny1 = min(mx1, ix1), min(my1, iy1)
                nx2, ny2 = max(mx2, ix2), max(my2, iy2)
                merged_bbox = [nx1, ny1, nx2 - nx1, ny2 - ny1]
                # Weighted average confidence by area
                m_area = np.count_nonzero(merged_mask > 0)
                i_area = np.count_nonzero(item["mask"] > 0)
                total_area = m_area + i_area
                merged_conf = (merged_conf * m_area + item["confidence"] * i_area) / total_area if total_area > 0 else merged_conf
            else:
                instance_count += 1
                rejected_count += 1

        quality = _mask_quality(merged_mask, merged_conf)
        if not _passes_quality_gate(quality, part_key):
            logger.debug("%s mask failed quality gate: %s", part_key, quality)
            continue

        result[part_key] = {
            "mask": merged_mask,
            "confidence": round(float(merged_conf), 3),
            "bbox": tuple(merged_bbox),
            "class_name": items[0]["class_name"],
            "quality": quality,
            "instance_count": instance_count,
            "rejected_count": rejected_count,
            "accepted_clusters": accepted_clusters,
        }
        accepted_masks.append(merged_mask)

    # Build whole mask from accepted parts
    if accepted_masks:
        whole = np.zeros((H, W), dtype=np.uint8)
        for m in accepted_masks:
            whole = np.maximum(whole, m)
        result["whole"] = {
            "mask": whole,
            "confidence": 1.0,
            "bbox": _bbox_from_mask(whole),
            "class_name": "Whole",
            "quality": _mask_quality(whole, 1.0),
            "instance_count": len(accepted_masks),
            "rejected_count": 0,
            "accepted_clusters": [],
        }

    return result


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def _bbox_to_corners(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    return x, y, x + w, y + h
