"""
Mushroom segmentation wrapper (YOLOv8-seg friendly).

Provides a lazy, thread-safe loader and a stable repo-local output contract.
If Ultralytics/YOLO is not installed, the module degrades gracefully and
`get_segmenter()` will raise ImportError on first use.
"""

from __future__ import annotations

import io
import threading
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


class Segmenter:
    def __init__(self, model_path: str):
        if YOLO is None:
            raise ImportError("Ultralytics YOLO is not installed; cannot load segmenter")
        self.model = YOLO(model_path)

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
        # Normalize into a list of instances with mask, bbox, confidence.
        instances: List[Dict[str, Any]] = []
        H, W = image_shape[:2]
        # results may be a Results or list-like; iterate robustly
        for r in results:
            # Some result objects expose `.masks` and `.boxes` depending on API.
            masks = getattr(r, "masks", None)
            boxes = getattr(r, "boxes", None)
            if masks is not None:
                # masks.data may be a (N, H, W) array or provide a .numpy() API
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
                        # attempt to find a confidence for this mask
                        conf = 0.0
                        # try boxes match
                        if boxes is not None:
                            try:
                                conf = float(boxes.data[i, 4].cpu().numpy())
                            except Exception:
                                conf = float(getattr(boxes[i], "confidence", 0.0) if hasattr(boxes, "__len__") else 0.0)
                        instances.append({"mask": m, "bbox": None, "model_confidence": conf})
            elif boxes is not None:
                # fallback: turn boxes into plain bboxes without masks
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
                        instances.append({"mask": m, "bbox": (x, y, w, h), "model_confidence": float(conf)})
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
            # normalize mask dtype
            if m.dtype != np.uint8:
                m = (m > 0).astype(np.uint8) * 255
            inst["mask"] = m
            inst["area_ratio"] = float(np.count_nonzero(m > 0) / (H * W))
            # cleanup candidate masks lightly
            cleaned = self._cleanup_mask(m)
            inst["cleaned_mask"] = cleaned
            inst.update(self._quality_metrics(cleaned))

        # sort by confidence then area
        instances = sorted(instances, key=lambda x: (x.get("model_confidence", 0.0), x.get("area_ratio", 0.0)), reverse=True)

        # limit to top_n
        instances = instances[:top_n]

        # selection: deterministic rules collectivized here but caller may re-rank
        selected_index = None
        for idx, inst in enumerate(instances):
            if inst.get("model_confidence", 0.0) < 0.0:
                continue
            selected_index = 0 if idx == 0 else selected_index
        # default selection if any
        if instances and selected_index is None:
            selected_index = 0

        return {
            "instances": instances,
            "selected_index": selected_index,
        }


def get_segmenter(model_path: str = "artifacts/yolov8_seg.pt") -> Segmenter:
    global _segmenter_instance
    if _segmenter_instance is not None:
        return _segmenter_instance
    with _segmenter_lock:
        if _segmenter_instance is None:
            _segmenter_instance = Segmenter(model_path)
    return _segmenter_instance
