"""
Mushroom segmentation wrapper (YOLOv8-seg friendly).

Provides a lazy, thread-safe loader and a stable repo-local output contract.
If Ultralytics/YOLO is not installed, the module degrades gracefully and
`get_segmenter()` will raise ImportError on first use.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Try to import Ultralytics YOLO if available. Keep optional to avoid hard
# dependency during local analysis or CI where the package may be absent.
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore

# Thread-safe singleton
_segmenter_lock = threading.Lock()
_segmenter_instance: Optional["Segmenter"] = None


# Class IDs from Roboflow 3-class segmentation
# 0=Cap, 1=Stem, 2=Underside
CLASS_NAMES = {
    0: "Cap",
    1: "Stem",
    2: "Underside",
}


class Segmenter:
    def __init__(self, model_path: str):
        if YOLO is None:
            raise ImportError("Ultralytics YOLO is not installed; cannot load segmenter")
        self.model = YOLO(model_path)
        # Cache model class names if available
        self._model_names = getattr(self.model, "names", CLASS_NAMES)

    def _pil_to_bgr(self, image_bytes: bytes) -> np.ndarray:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _model_predict(self, bgr: np.ndarray) -> Any:
        # Ultralytics YOLO v8 API may be invoked as `self.model(bgr)` returning
        # a Results object. Keep this flexible.
        try:
            res = self.model(bgr)
            return res
        except Exception:
            # Try predict alias
            try:
                res = self.model.predict(bgr)
                return res
            except Exception as exc:
                raise RuntimeError(f"YOLO model inference failed: {exc}")

    def _parse_results(self, results: Any, image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        # Normalize into a list of instances with class, mask, bbox, confidence.
        instances: List[Dict[str, Any]] = []
        H, W = image_shape[:2]
        for r in results:
            masks = getattr(r, "masks", None)
            boxes = getattr(r, "boxes", None)
            # Extract class IDs when available
            class_ids = []
            if boxes is not None:
                try:
                    cls_tensor = getattr(boxes, "cls", None)
                    if cls_tensor is not None:
                        class_ids = cls_tensor.cpu().numpy().astype(int).tolist()
                except Exception:
                    pass

            if masks is not None:
                try:
                    arr = masks.data.cpu().numpy()
                except Exception:
                    try:
                        arr = np.asarray(masks)
                    except Exception:
                        arr = None
                if arr is not None:
                    for i in range(arr.shape[0]):
                        m = (arr[i] * 255).astype(np.uint8) if arr.dtype.kind == "f" else arr[i].astype(np.uint8)
                        # Resize mask to original image dimensions if needed
                        if m.shape[0] != H or m.shape[1] != W:
                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                        conf = 0.0
                        cls_id = class_ids[i] if i < len(class_ids) else -1
                        if boxes is not None:
                            try:
                                conf = float(boxes.data[i, 4].cpu().numpy())
                            except Exception:
                                conf = float(getattr(boxes[i], "confidence", 0.0) if hasattr(boxes, "__len__") else 0.0)
                        instances.append({
                            "class_id": cls_id,
                            "class_name": self._model_names.get(cls_id, CLASS_NAMES.get(cls_id, "unknown")),
                            "mask": m,
                            "bbox": None,
                            "model_confidence": conf,
                        })
            elif boxes is not None:
                try:
                    arr = boxes.data.cpu().numpy()
                except Exception:
                    try:
                        arr = np.asarray(boxes)
                    except Exception:
                        arr = None
                if arr is not None:
                    for i in range(arr.shape[0]):
                        x1, y1, x2, y2, conf = arr[i][:5]
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        m = np.zeros((H, W), dtype=np.uint8)
                        cv2.rectangle(m, (x, y), (x + w, y + h), 255, -1)
                        cls_id = class_ids[i] if i < len(class_ids) else -1
                        instances.append({
                            "class_id": cls_id,
                            "class_name": self._model_names.get(cls_id, CLASS_NAMES.get(cls_id, "unknown")),
                            "mask": m,
                            "bbox": (x, y, w, h),
                            "model_confidence": float(conf),
                        })
        return instances

    def _bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return 0, 0, 0, 0
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        return x1, y1, x2 - x1 + 1, y2 - y1 + 1

    def _cleanup_mask(self, mask: np.ndarray, min_area: int = 64, morph_iter: int = 1) -> np.ndarray:
        # Ensure binary 0/255 uint8
        mask_u = (mask > 0).astype(np.uint8) * 255
        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u, connectivity=8)
        if num_labels <= 1:
            cleaned = mask_u
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.where(areas >= min_area)[0] + 1
            cleaned = np.zeros_like(mask_u)
            for lab in keep:
                cleaned[labels == lab] = 255
        # Morphological close/open
        if morph_iter > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
        return cleaned

    def _quality_metrics(self, mask: np.ndarray) -> Dict[str, Any]:
        H, W = mask.shape[:2]
        area = float(np.count_nonzero(mask > 0))
        area_ratio = area / float(H * W)
        # contours and hole estimation
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        frag = 0
        hole_ratio = 0.0
        boundary_irregularity = 0.0
        if contours:
            frag = len(contours)
            perimeters = [cv2.arcLength(c, True) for c in contours]
            perim = sum(perimeters) if perimeters else 0.0
            areas = [cv2.contourArea(c) for c in contours]
            area_sum = sum(areas) if areas else 0.0
            boundary_irregularity = (perim / area_sum) if area_sum > 0 else 0.0
            # hole ratio: number of inner contours / outer contours approx
            if hierarchy is not None:
                # hierarchy shape (N, 4)
                hier = hierarchy[0]
                inner = sum(1 for h in hier if h[3] != -1)
                hole_ratio = float(inner) / float(len(hier)) if len(hier) > 0 else 0.0
        return {
            "area_ratio": area_ratio,
            "fragment_count": int(frag),
            "hole_ratio": float(hole_ratio),
            "boundary_irregularity": float(boundary_irregularity),
        }

    def _center_distance(self, bbox: Tuple[int, ...], W: int, H: int) -> float:
        """Normalized distance from bbox centroid to image center."""
        x, y, bw, bh = bbox
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        return max(abs(cx - W / 2.0) / W, abs(cy - H / 2.0) / H)

    def _skin_ratio(self, bgr: np.ndarray, mask: np.ndarray) -> float:
        """Fraction of masked pixels in HSV skin-tone range."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Skin-tone range (H: 0-50, S: 20-170, V: 50-255)
        lower = np.array([0, 20, 50], dtype=np.uint8)
        upper = np.array([50, 170, 255], dtype=np.uint8)
        skin = cv2.inRange(hsv, lower, upper)
        masked_skin = np.count_nonzero((mask > 0) & (skin > 0))
        mask_pixels = np.count_nonzero(mask > 0)
        return masked_skin / mask_pixels if mask_pixels > 0 else 0.0

    def _aspect_ratio(self, bbox: Tuple[int, ...]) -> float:
        x, y, w, h = bbox
        h = max(1, h)
        return w / h

    def segment(self, image_bytes: bytes, top_n: int = 5) -> Dict[str, Any]:
        bgr = self._pil_to_bgr(image_bytes)
        H, W = bgr.shape[:2]
        results = self._model_predict(bgr)
        instances = self._parse_results(results, (H, W))

        # populate bbox and metrics for each instance
        for inst in instances:
            if inst.get("bbox") is None:
                inst["bbox"] = self._bbox_from_mask(inst["mask"]) if inst.get("mask") is not None else (0, 0, 0, 0)
            m = inst.get("mask", np.zeros((H, W), dtype=np.uint8))
            if m.dtype != np.uint8:
                m = (m > 0).astype(np.uint8) * 255
            inst["mask"] = m
            inst["area_ratio"] = float(np.count_nonzero(m > 0) / (H * W))
            cleaned = self._cleanup_mask(m)
            inst["cleaned_mask"] = cleaned
            inst.update(self._quality_metrics(cleaned))

            bbox = inst["bbox"]
            inst["center_distance"] = self._center_distance(bbox, W, H)
            inst["skin_ratio"] = self._skin_ratio(bgr, cleaned)
            inst["aspect_ratio"] = self._aspect_ratio(bbox)

        # Filter + rank
        filtered: List[Dict[str, Any]] = []
        for inst in instances:
            ar = inst.get("aspect_ratio", 1.0)
            if ar > 4.0 or ar < 0.25:
                continue
            if inst.get("skin_ratio", 0.0) > 0.30:
                continue
            filtered.append(inst)

        if not filtered:
            filtered = instances

        filtered.sort(
            key=lambda x: (x.get("model_confidence", 0.0), -x.get("center_distance", 1.0)),
            reverse=True,
        )
        filtered = filtered[:top_n]
        selected_index = 0 if filtered else None

        return {
            "instances": filtered,
            "selected_index": selected_index,
        }


def detect_case_from_masks(
    above_masks: Dict[str, Dict[str, Any]],
    below_masks: Dict[str, Dict[str, Any]],
    confidence_threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Detect morphological case from *filtered* part masks (output of
    :func:`build_part_masks`).

    Returns:
        {
            "case": "classical" | "puffball" | "uncertain",
            "confidence": float,
            "detected_parts": ["Cap", "Stem", ...],
            "reasoning": str,
        }
    """
    above_parts = set(above_masks.keys())
    below_parts = set(below_masks.keys())
    all_parts = above_parts | below_parts

    has_cap = "cap" in all_parts
    has_stem = "stem" in all_parts
    has_underside = "underside" in all_parts

    # Classical: cap + stem/underside, OR stem + underside
    if (has_cap and (has_stem or has_underside)) or (has_stem and has_underside):
        confidence = 0.80 if has_cap else 0.65
        return {
            "case": "classical",
            "confidence": confidence,
            "detected_parts": sorted([p.capitalize() for p in all_parts if p != "whole"]),
            "reasoning": "Classical mushroom morphology: stem and underside visible." if not has_cap else "Classical mushroom morphology: cap with stem/underside visible.",
        }

    # Puffball: cap-only in at least one photo, no stem/underside anywhere
    if has_cap and not has_stem and not has_underside:
        return {
            "case": "puffball",
            "confidence": 0.60,
            "detected_parts": sorted([p.capitalize() for p in all_parts if p != "whole"]),
            "reasoning": "Only cap detected; no stem or underside visible. Likely puffball or ball-shaped fungus.",
        }

    return {
        "case": "uncertain",
        "confidence": 0.0,
        "detected_parts": sorted([p.capitalize() for p in all_parts if p != "whole"]),
        "reasoning": "Insufficient morphological information to determine case.",
    }


def detect_case(
    above_instances: List[Dict[str, Any]],
    below_instances: List[Dict[str, Any]],
    confidence_threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Detect morphological case from YOLO segmentation outputs of above/below photos.

    Rules (from implementation plan):
      - Cap + (Stem or Underside) in either photo → "classical"
      - Cap-only in BOTH photos (no underside/stem) → "puffball"
      - Anything else → "uncertain"

    Returns:
      {
        "case": "classical" | "puffball" | "uncertain",
        "confidence": float,
        "detected_parts": ["Cap", "Stem", ...],
        "reasoning": str,
      }
    """
    all_instances = list(above_instances) + list(below_instances)

    # Collect parts that pass confidence threshold
    parts = set()
    for inst in all_instances:
        conf = inst.get("model_confidence", 0.0)
        cls_name = inst.get("class_name", "unknown")
        if conf >= confidence_threshold and cls_name != "unknown":
            parts.add(cls_name)

    has_cap = "Cap" in parts
    has_stem = "Stem" in parts
    has_underside = "Underside" in parts

    # Classical: cap + stem/underside, OR stem + underside (cap may be occluded/filtered)
    if (has_cap and (has_stem or has_underside)) or (has_stem and has_underside):
        confidence = 0.80 if has_cap else 0.65
        return {
            "case": "classical",
            "confidence": confidence,
            "detected_parts": sorted(parts),
            "reasoning": "Classical mushroom morphology: stem and underside visible." if not has_cap else "Classical mushroom morphology: cap with stem/underside visible.",
        }

    # Puffball: cap-only in at least one photo, no stem/underside anywhere
    if has_cap and not has_stem and not has_underside:
        return {
            "case": "puffball",
            "confidence": 0.60,
            "detected_parts": sorted(parts),
            "reasoning": "Only cap detected; no stem or underside visible. Likely puffball or ball-shaped fungus.",
        }

    return {
        "case": "uncertain",
        "confidence": 0.0,
        "detected_parts": sorted(parts),
        "reasoning": "Insufficient morphological information to determine case.",
    }


def _resolve_model_path(
    preferred: str = "data/Yolov8/best.pt",
    fallback: str = "yolov8n-seg.pt",
) -> str:
    """Return preferred path if it exists, otherwise fallback."""
    if Path(preferred).exists():
        return preferred
    if Path(fallback).exists():
        return fallback
    # Let Ultralytics handle download if neither exists
    return preferred if preferred else fallback


def get_segmenter(
    model_path: Optional[str] = None,
    preferred_path: str = "data/Yolov8/best.pt",
    fallback_path: str = "yolov8n-seg.pt",
) -> Segmenter:
    global _segmenter_instance
    if _segmenter_instance is not None:
        return _segmenter_instance
    with _segmenter_lock:
        if _segmenter_instance is None:
            resolved = model_path if model_path else _resolve_model_path(preferred_path, fallback_path)
            _segmenter_instance = Segmenter(resolved)
    return _segmenter_instance
