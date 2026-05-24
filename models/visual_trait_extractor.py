"""
Visual Trait Extractor — Step 1 of the mushroom identification pipeline.

Analyses a raw image (optionally constrained by segmentation part masks) using
classical computer vision to produce a structured dictionary of visible
traits.  The extractor is intentionally **pure**: it describes what it sees
(colour, shape, texture, brightness) but never attempts to identify the
species.  Species classification is left to downstream components (CNN hint,
trait-database comparator, key-tree traversal, LLM, and final aggregator).

If a trained CNN is available, its top-k output is returned as an optional
``ml_prediction`` hint, but this is produced by the CNN model — not by the
trait extractor itself.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour vocabulary
# ---------------------------------------------------------------------------

# (name, hue_min, hue_max, sat_min, val_min) — OpenCV H is 0-179
_COLOUR_RULES: List[Tuple[str, int, int, int, int]] = [
    ("red",          0,  10, 80, 60),
    ("red",        160, 179, 80, 60),   # wraps around 180
    ("orange",      10,  25, 80, 80),
    ("yellow",      25,  35, 70, 80),
    ("orange-yellow", 15, 35, 70, 80),  # broad band used for scoring
    ("yellow-green", 35,  70, 60, 60),
    ("green",        70,  85, 60, 50),
    ("olive-brown",  10,  30, 40, 40),
    ("brown",        10,  25, 40, 30),
    ("tan",          15,  30, 25, 90),
    ("white",         0, 179,  0, 200),
    ("grey",          0, 179,  0, 80),
    ("black",         0, 179,  0, 30),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_hsv(pixels: np.ndarray, n_clusters: int = 4) -> List[Tuple[float, float, float]]:
    """Return HSV cluster centres sorted by cluster size (largest first)."""
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    km.fit(pixels)
    centres = km.cluster_centers_
    counts = np.bincount(km.labels_)
    order = np.argsort(counts)[::-1]
    return [tuple(centres[i]) for i in order]   # type: ignore[return-value]


def _hsv_to_name(h: float, s: float, v: float) -> str:
    """Map a single HSV triplet to a human-readable colour name."""
    if v < 35:
        return "black"
    if s < 25 and v > 190:
        return "white"
    if s < 40 and v < 120:
        return "grey"

    for name, h_lo, h_hi, s_min, v_min in _COLOUR_RULES:
        if h_lo <= h <= h_hi and s >= s_min and v >= v_min:
            return name

    # fallback by hue band
    if h < 15 or h > 165:
        return "red"
    if h < 30:
        return "orange"
    if h < 40:
        return "yellow"
    if h < 75:
        return "green"
    if h < 135:
        return "blue-grey"
    return "brown"


def _bgr_pixels_to_hsv(pixels: np.ndarray) -> np.ndarray:
    """Convert a (N, 3) BGR pixel array to (N, 3) HSV using OpenCV."""
    N = pixels.shape[0]
    img_3d = pixels.reshape(1, N, 3).astype(np.uint8)
    hsv_3d = cv2.cvtColor(img_3d, cv2.COLOR_BGR2HSV)
    return hsv_3d.reshape(N, 3).astype(np.float32)


def _sample_pixels_masked(bgr: np.ndarray, mask: np.ndarray, max_pixels: int = 4096) -> np.ndarray:
    """Deterministically sample masked pixels for KMeans."""
    mask_bool = mask > 0
    if mask_bool.sum() < 50:
        return np.array([])
    pixels = bgr[mask_bool]
    if len(pixels) > max_pixels:
        step = max(1, len(pixels) // max_pixels)
        pixels = pixels[::step][:max_pixels]
    return pixels


def _contour_metrics(mask: np.ndarray) -> Dict[str, float]:
    """Compute contour-based shape metrics from a binary mask."""
    mask_u = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "aspect_ratio": 1.0,
            "circularity": 0.5,
            "convexity": 1.0,
            "solidity": 1.0,
            "central_depression_score": 0.0,
            "contour_complexity": 0.0,
        }

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / max(h, 1)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    convexity = hull_area / area if area > 0 else 1.0
    solidity = area / hull_area if hull_area > 0 else 1.0

    # Central depression: compare center pixel region area to convex hull center
    center_mask = np.zeros_like(mask_u)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(center_mask, (cx, cy), min(w, h) // 4, 255, -1)
    center_area = cv2.countNonZero(cv2.bitwise_and(mask_u, center_mask))
    circle_area = np.pi * (min(w, h) // 4) ** 2
    central_depression_score = 1.0 - (center_area / circle_area) if circle_area > 0 else 0.0

    # Contour complexity: perimeter² / area (normalized)
    contour_complexity = (perimeter ** 2) / (4 * np.pi * area) - 1.0 if area > 0 else 0.0
    contour_complexity = max(0.0, min(contour_complexity, 10.0))

    return {
        "aspect_ratio": round(float(aspect_ratio), 2),
        "circularity": round(float(circularity), 2),
        "convexity": round(float(convexity), 2),
        "solidity": round(float(solidity), 2),
        "central_depression_score": round(float(central_depression_score), 2),
        "contour_complexity": round(float(contour_complexity), 2),
    }


def _classify_cap_shape(metrics: Dict[str, float]) -> str:
    """Mutually exclusive cap-shape decision tree."""
    ar = metrics["aspect_ratio"]
    circ = metrics["circularity"]
    dep = metrics["central_depression_score"]
    comp = metrics["contour_complexity"]

    if circ > 0.80 and 0.8 <= ar <= 1.3:
        return "convex"
    elif ar >= 1.6 and circ < 0.6:
        return "flat"
    elif ar <= 0.5 and circ >= 0.70:
        return "bell-shaped"
    elif 0.5 < ar < 0.9 and circ < 0.60:
        return "funnel-shaped"
    elif dep > 0.6 and ar >= 0.9:
        return "depressed"
    elif ar >= 0.9 and circ < 0.45:
        return "wavy"
    elif comp > 0.8 and circ < 0.40:
        return "irregular"
    elif ar < 0.9 and circ >= 0.60:
        return "bell-shaped"
    else:
        return "unknown"


def _compute_colour_ratios(pixels: np.ndarray) -> Dict[str, float]:
    """Compute colour ratios from a (N, 3) BGR pixel array."""
    if len(pixels) == 0:
        return {"red": 0.0, "orange_red": 0.0, "orange_yellow": 0.0,
                "brown": 0.0, "white": 0.0, "dark": 0.0}
    arr = pixels.astype(np.float32)
    r, g, b = arr[:, 2], arr[:, 1], arr[:, 0]
    return {
        "red": round(float(np.mean((r > 150) & (r > g * 1.25) & (r > b * 1.25))), 3),
        "orange_red": round(float(np.mean((r > 140) & (g > 50) & (g < 130) & (b < 80))), 3),
        "orange_yellow": round(float(np.mean((r > 140) & (g > 90) & (b < 140))), 3),
        "brown": round(float(np.mean((r > 80)  & (g > 45) & (b < 90) & (r > g) & (g > b))), 3),
        "white": round(float(np.mean((r > 185) & (g > 185) & (b > 185))), 3),
        "dark": round(float(np.mean((r < 60)  & (g < 60)  & (b < 60))), 3),
    }


def _compute_texture_metrics(bgr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Compute per-mask texture metrics with correct mask-area denominator."""
    mask_bool = mask > 0
    if mask_bool.sum() < 50:
        return {"surface_texture": "unknown", "edge_density": 0.0, "has_ridges": False}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges[~mask_bool] = 0

    mask_area = int(np.count_nonzero(mask_bool))
    edge_density = float(np.count_nonzero(edges > 0) / max(mask_area, 1))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=5)
    has_ridges = lines is not None and len(lines) > 10

    if edge_density < 0.05:
        texture = "smooth"
    elif edge_density < 0.15:
        texture = "fibrous"
    else:
        texture = "scaly"

    return {"surface_texture": texture, "edge_density": round(edge_density, 3), "has_ridges": bool(has_ridges)}


def trait_confidence(mask_quality: Dict[str, Any], detector_score: float) -> float:
    """
    Compute a confidence score for a part-aware trait.
    mask_quality keys: area_ratio, fragmentation, hole_ratio,
                       boundary_irregularity, yolo_confidence, touches_border
    """
    area_q = min(mask_quality.get("area_ratio", 0.0) / 0.05, 1.0)
    frag_q = max(0.0, 1.0 - (mask_quality.get("fragmentation", 1) - 1) * 0.25)
    hole_q = max(0.0, 1.0 - mask_quality.get("hole_ratio", 0.0) / 0.10)
    border_penalty = 0.9 if mask_quality.get("touches_border", False) else 1.0
    yolo_conf = mask_quality.get("yolo_confidence", 0.0)
    return float(area_q * frag_q * hole_q * border_penalty * yolo_conf * detector_score)


# ---------------------------------------------------------------------------
# Legacy whole-image analysers (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def analyse_colours(bgr: np.ndarray) -> Dict[str, Any]:
    """
    Extract colour profile from image.
    """
    small = cv2.resize(bgr, (128, 128))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    clusters = _dominant_hsv(hsv, n_clusters=5)
    colour_names = [_hsv_to_name(*c) for c in clusters]

    dominant = colour_names[0] if colour_names else "unknown"
    secondary = colour_names[1] if len(colour_names) > 1 else dominant

    ratios = _compute_colour_ratios(small.reshape(-1, 3))
    ratios["dominant_color"] = dominant
    ratios["secondary_color"] = secondary
    return ratios


def analyse_shape(bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimate cap shape from the image contour.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"cap_shape": "unknown", "aspect_ratio": 1.0, "circularity": 0.5}

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    x, y, w, h = cv2.boundingRect(largest)

    aspect_ratio = w / max(h, 1)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    if circularity > 0.80 and 0.8 < aspect_ratio < 1.3:
        cap_shape = "convex"
    elif aspect_ratio > 1.6 and circularity < 0.6:
        cap_shape = "flat"
    elif aspect_ratio < 0.7:
        cap_shape = "bell-shaped"
    elif circularity < 0.45:
        cap_shape = "wavy"
    elif 0.5 < aspect_ratio < 1.0 and circularity < 0.65:
        cap_shape = "funnel-shaped"
    else:
        cap_shape = "convex"

    return {"cap_shape": cap_shape, "aspect_ratio": round(float(aspect_ratio), 2), "circularity": round(float(circularity), 2)}


def analyse_texture(bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimate surface texture via edge density.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=20, maxLineGap=5)
    has_ridges = lines is not None and len(lines) > 15

    if edge_density < 0.05:
        texture = "smooth"
    elif edge_density < 0.15:
        texture = "fibrous"
    else:
        texture = "scaly"

    return {"surface_texture": texture, "edge_density": round(edge_density, 3), "has_ridges": bool(has_ridges)}


def analyse_brightness(bgr: np.ndarray) -> str:
    """Return 'dark', 'medium', or 'bright' based on mean value channel."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_v = float(np.mean(hsv[:, :, 2]))
    if mean_v < 70:
        return "dark"
    if mean_v < 160:
        return "medium"
    return "bright"


# ---------------------------------------------------------------------------
# Fixed masked analysers
# ---------------------------------------------------------------------------

def analyse_colours_masked(bgr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Compute colour stats only on mask-positive pixels. Fall back to full-image if mask too small."""
    pixels = _sample_pixels_masked(bgr, mask, max_pixels=4096)
    if len(pixels) == 0:
        return analyse_colours(bgr)

    hsv = _bgr_pixels_to_hsv(pixels)
    clusters = _dominant_hsv(hsv, n_clusters=min(4, len(pixels)))
    colour_names = [_hsv_to_name(*c) for c in clusters]
    dominant = colour_names[0] if colour_names else "unknown"
    secondary = colour_names[1] if len(colour_names) > 1 else dominant

    ratios = _compute_colour_ratios(pixels)
    ratios["dominant_color"] = dominant
    ratios["secondary_color"] = secondary
    return ratios


def analyse_texture_masked(bgr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Fixed texture analysis with mask-area denominator."""
    return _compute_texture_metrics(bgr, mask)


def analyse_brightness_masked(bgr: np.ndarray, mask: np.ndarray) -> str:
    mask_bool = mask > 0
    if mask_bool.sum() < 10:
        return analyse_brightness(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_v = float(np.mean(hsv[:, :, 2][mask_bool]))
    if mean_v < 70:
        return "dark"
    if mean_v < 160:
        return "medium"
    return "bright"


def analyse_shape_masked(bgr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Use mask contour as primary source; fallback to analyse_shape when ambiguous."""
    metrics = _contour_metrics(mask)
    cap_shape = _classify_cap_shape(metrics)
    metrics["cap_shape"] = cap_shape
    return metrics


# ---------------------------------------------------------------------------
# Part-specific trait analysers
# ---------------------------------------------------------------------------

def analyse_cap_traits(bgr: np.ndarray, cap_mask: np.ndarray) -> Dict[str, Any]:
    """Extract cap-specific traits from the cap mask."""
    colour = analyse_colours_masked(bgr, cap_mask)
    shape_metrics = analyse_shape_masked(bgr, cap_mask)
    texture = analyse_texture_masked(bgr, cap_mask)
    brightness = analyse_brightness_masked(bgr, cap_mask)

    cap_surface = texture["surface_texture"]
    # Refine surface categories for cap
    if texture["edge_density"] > 0.25:
        cap_surface = "scaly"
    elif texture["edge_density"] > 0.18:
        cap_surface = "coarse-scaly"

    return {
        "cap_color": colour["dominant_color"],
        "cap_secondary_color": colour["secondary_color"],
        "cap_shape": shape_metrics["cap_shape"],
        "cap_surface": cap_surface,
        "cap_brightness": brightness,
        "cap_colour_ratios": {
            "red": colour["red"],
            "orange_red": colour.get("orange_red", 0.0),
            "orange_yellow": colour["orange_yellow"],
            "brown": colour["brown"],
            "white": colour["white"],
            "dark": colour["dark"],
        },
        "cap_aspect_ratio": shape_metrics.get("aspect_ratio"),
        "cap_circularity": shape_metrics.get("circularity"),
    }


def analyse_stem_traits(bgr: np.ndarray, stem_mask: np.ndarray) -> Dict[str, Any]:
    """Extract stem-specific traits from the stem mask."""
    colour = analyse_colours_masked(bgr, stem_mask)
    texture = analyse_texture_masked(bgr, stem_mask)
    brightness = analyse_brightness_masked(bgr, stem_mask)

    # Basic thickness from bbox
    ys, xs = np.where(stem_mask > 0)
    thickness = float(xs.max() - xs.min()) if len(xs) > 0 else 0.0
    height = float(ys.max() - ys.min()) if len(ys) > 0 else 0.0
    aspect_ratio = thickness / max(height, 1)

    # Ring detection: look for horizontal band of high contrast in upper half of stem
    stem_ring = "unknown"
    if len(ys) > 0 and len(xs) > 0:
        y_mid = int(ys.min() + (ys.max() - ys.min()) * 0.3)
        ring_region = stem_mask[ys.min():y_mid, xs.min():xs.max()]
        if ring_region.size > 0:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ring_gray = gray[ys.min():y_mid, xs.min():xs.max()]
            # Simple ring heuristic: high std in upper region suggests annulus
            if np.std(ring_gray) > 40:
                stem_ring = "present"
            else:
                stem_ring = "absent"

    stem_surface = texture["surface_texture"]
    if texture["edge_density"] > 0.20:
        stem_surface = "fibrous"

    return {
        "stem_color": colour["dominant_color"],
        "stem_brightness": brightness,
        "stem_thickness": round(thickness, 1),
        "stem_aspect_ratio": round(aspect_ratio, 2),
        "stem_ring": stem_ring,
        "stem_surface": stem_surface,
    }


def analyse_underside_traits(
    bgr: np.ndarray,
    underside_mask: np.ndarray,
    stem_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Extract underside-specific traits: gills, pores, ridges, teeth."""
    colour = analyse_colours_masked(bgr, underside_mask)
    texture = analyse_texture_masked(bgr, underside_mask)

    mask_bool = underside_mask > 0
    if mask_bool.sum() < 50:
        return {
            "underside_color": colour["dominant_color"],
            "hymenophore_type": "unknown",
            "hymenophore_confidence": 0.0,
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges[~mask_bool] = 0

    # Hough lines for gills / ridges
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=3)
    line_count = len(lines) if lines is not None else 0

    # Blob detection for pores
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect((underside_mask > 0).astype(np.uint8) * 255)
    blob_count = len(keypoints)

    # Heuristic scoring
    gill_score = min(line_count / 30.0, 1.0) if line_count > 15 else 0.0
    pore_score = min(blob_count / 50.0, 1.0) if blob_count > 10 else 0.0
    ridge_score = min(line_count / 20.0, 1.0) if 8 < line_count <= 15 else 0.0
    tooth_score = min(texture["edge_density"] / 0.3, 1.0) if texture["edge_density"] > 0.2 else 0.0

    scores = {
        "gills": gill_score,
        "pores": pore_score,
        "ridges": ridge_score,
        "teeth": tooth_score,
    }
    best = max(scores, key=scores.get)
    best_score = scores[best]
    second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0

    confidence = best_score
    if best_score - second_best < 0.3:
        confidence *= 0.5

    if confidence < 0.2:
        hymenophore_type = "unknown"
    else:
        hymenophore_type = best

    return {
        "underside_color": colour["dominant_color"],
        "hymenophore_type": hymenophore_type,
        "hymenophore_confidence": round(confidence, 3),
    }


def analyse_puffball_traits(bgr: np.ndarray, whole_mask: np.ndarray) -> Dict[str, Any]:
    """Extract puffball-specific traits from the whole-body mask."""
    colour = analyse_colours_masked(bgr, whole_mask)
    texture = analyse_texture_masked(bgr, whole_mask)
    brightness = analyse_brightness_masked(bgr, whole_mask)
    metrics = _contour_metrics(whole_mask)

    # Surface classification
    puffball_surface = "smooth"
    if texture["edge_density"] > 0.15:
        puffball_surface = "warty"
    elif texture["edge_density"] > 0.25:
        puffball_surface = "spiny"

    return {
        "whole_color": colour["dominant_color"],
        "puffball_surface": puffball_surface,
        "puffball_roundness": metrics.get("circularity"),
        "puffball_aspect_ratio": metrics.get("aspect_ratio"),
        "puffball_brightness": brightness,
    }


# ---------------------------------------------------------------------------
# Morphology case derivation
# ---------------------------------------------------------------------------

def is_puffball_like(detected_parts: set, whole_shape_metrics: Dict[str, float]) -> bool:
    return (
        detected_parts == {"cap"}
        and whole_shape_metrics.get("circularity", 0.0) > 0.75
        and whole_shape_metrics.get("aspect_ratio", 1.0) < 1.3
    )


def derive_morphology_case(
    detected_parts: set,
    cap_shape: str,
    whole_shape_metrics: Dict[str, float],
) -> str:
    if is_puffball_like(detected_parts, whole_shape_metrics):
        return "puffball"
    if "cap" in detected_parts and ("stem" in detected_parts or "underside" in detected_parts):
        if cap_shape in {"funnel-shaped", "depressed"}:
            return "classical_concave"
        if cap_shape in {"convex", "flat", "bell-shaped"}:
            return "classical_convex"
        return "classical_unknown"
    return "uncertain"


# ---------------------------------------------------------------------------
# Part-aware extraction path
# ---------------------------------------------------------------------------

def _build_part_masks_from_segmentation(image_bytes: bytes) -> Optional[Dict[str, Dict[str, Any]]]:
    """Run YOLO segmentation and build YoloPartMasks."""
    try:
        from models.mushroom_segmenter import get_segmenter
        from models.yolo_part_masks import build_part_masks
        seg = get_segmenter()
        seg_res = seg.segment(image_bytes)
        instances = seg_res.get("instances", [])
        if not instances:
            return None
        # Need image shape — decode once
        import io
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        H, W = np.array(pil_img).shape[:2]
        return build_part_masks(instances, (H, W))
    except Exception as exc:
        logger.debug("Failed to build part masks from segmentation: %s", exc)
        return None


def _part_aware_extract(
    image_bytes: bytes,
    part_masks: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """New part-aware extraction path."""
    import io as _io

    pil_img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Build part masks if not provided
    if part_masks is None:
        part_masks = _build_part_masks_from_segmentation(image_bytes)

    if not part_masks:
        # No valid part masks — fall back to whole-image classical analysis
        colour = analyse_colours(bgr)
        shape = analyse_shape(bgr)
        texture = analyse_texture(bgr)
        brightness = analyse_brightness(bgr)
        visible_traits = {
            "dominant_color": colour["dominant_color"],
            "secondary_color": colour["secondary_color"],
            "cap_shape": shape["cap_shape"],
            "surface_texture": texture["surface_texture"],
            "has_ridges": texture["has_ridges"],
            "brightness": brightness,
            "colour_ratios": {
                "red": colour["red"],
                "orange_red": colour.get("orange_red", 0.0),
                "orange_yellow": colour["orange_yellow"],
                "brown": colour["brown"],
                "white": colour["white"],
                "dark": colour["dark"],
            },
            "mask_used": False,
            "morphology_case": "uncertain",
            "coarse_case": "uncertain",
            "detected_parts": [],
            # Fallback aliases for downstream consumers
            "cap_color": colour["dominant_color"],
            "whole_color": colour["dominant_color"],
            "stem_color": "unknown",
            "underside_color": "unknown",
            "cap_surface": texture["surface_texture"],
            "stem_surface": "unknown",
            "hymenophore_type": "unknown",
        }
        return {"visible_traits": visible_traits}

    # Collect detected part names
    detected_parts = set(part_masks.keys())
    detected_parts.discard("whole")
    detected_parts.discard("coral")

    # Run part-specific analysers
    traits_by_part: Dict[str, Dict[str, Any]] = {}
    trait_confidences: Dict[str, float] = {}
    trait_sources: Dict[str, str] = {}

    cap_mask_info = part_masks.get("cap")
    stem_mask_info = part_masks.get("stem")
    underside_mask_info = part_masks.get("underside")
    whole_mask_info = part_masks.get("whole")

    whole_metrics = _contour_metrics(whole_mask_info["mask"]) if whole_mask_info else {}

    # Cap traits
    if cap_mask_info:
        cap_traits = analyse_cap_traits(bgr, cap_mask_info["mask"])
        traits_by_part.update(cap_traits)
        for k in cap_traits:
            trait_confidences[k] = round(
                trait_confidence(cap_mask_info["quality"], 0.85), 3
            )
            trait_sources[k] = "yolo_cap_mask"

    # Stem traits
    if stem_mask_info:
        stem_traits = analyse_stem_traits(bgr, stem_mask_info["mask"])
        traits_by_part.update(stem_traits)
        for k in stem_traits:
            trait_confidences[k] = round(
                trait_confidence(stem_mask_info["quality"], 0.70), 3
            )
            trait_sources[k] = "yolo_stem_mask"

    # Underside traits
    if underside_mask_info:
        stem_mask = stem_mask_info["mask"] if stem_mask_info else None
        underside_traits = analyse_underside_traits(bgr, underside_mask_info["mask"], stem_mask)
        traits_by_part.update(underside_traits)
        for k in underside_traits:
            conf = underside_traits.get("hymenophore_confidence", 0.5)
            trait_confidences[k] = round(
                trait_confidence(underside_mask_info["quality"], conf), 3
            )
            trait_sources[k] = "yolo_underside_mask"

    # Puffball / whole traits
    if whole_mask_info:
        puffball_traits = analyse_puffball_traits(bgr, whole_mask_info["mask"])
        traits_by_part.update(puffball_traits)
        for k in puffball_traits:
            trait_confidences[k] = round(
                trait_confidence(whole_mask_info["quality"], 0.75), 3
            )
            trait_sources[k] = "yolo_whole_mask"

    # Derive morphology case
    cap_shape = traits_by_part.get("cap_shape", "unknown")
    morphology_case = derive_morphology_case(detected_parts, cap_shape, whole_metrics)

    # Determine coarse case
    if morphology_case == "puffball":
        coarse_case = "puffball"
    elif "cap" in detected_parts or "stem" in detected_parts or "underside" in detected_parts:
        coarse_case = "classical"
    else:
        coarse_case = "uncertain"

    # Clustered growth
    clustered_growth = any(
        info.get("instance_count", 1) > 1
        for key, info in part_masks.items()
        if key != "whole"
    )

    # Build compatibility keys
    cap_color = traits_by_part.get("cap_color", traits_by_part.get("whole_color", "unknown"))
    whole_color = traits_by_part.get("whole_color", cap_color)
    cap_surface = traits_by_part.get("cap_surface", "unknown")

    # Legacy compatibility mapping
    visible_traits: Dict[str, Any] = {
        "dominant_color": cap_color if cap_color != "unknown" else whole_color,
        "secondary_color": traits_by_part.get("cap_secondary_color", whole_color),
        "cap_shape": cap_shape,
        "surface_texture": cap_surface if cap_surface != "unknown" else traits_by_part.get("puffball_surface", "unknown"),
        "has_ridges": traits_by_part.get("hymenophore_type") == "ridges",
        "brightness": traits_by_part.get("cap_brightness", traits_by_part.get("puffball_brightness", "medium")),
        "colour_ratios": traits_by_part.get("cap_colour_ratios", {}),
        # New part-aware fields
        "morphology_case": morphology_case,
        "coarse_case": coarse_case,
        "detected_parts": sorted([p for p in detected_parts]),
        "hymenophore_type": traits_by_part.get("hymenophore_type", "unknown"),
        "cap_color": cap_color,
        "stem_color": traits_by_part.get("stem_color", "unknown"),
        "underside_color": traits_by_part.get("underside_color", "unknown"),
        "whole_color": whole_color,
        "cap_surface": cap_surface,
        "stem_ring": traits_by_part.get("stem_ring", "unknown"),
        "stem_surface": traits_by_part.get("stem_surface", "unknown"),
        "puffball_surface": traits_by_part.get("puffball_surface", "unknown"),
        "clustered_growth": clustered_growth,
        "trait_confidence": trait_confidences,
        "trait_source_by_key": trait_sources,
        "mask_used": True,
    }

    # Ensure colour_ratios is never empty
    if not visible_traits["colour_ratios"]:
        visible_traits["colour_ratios"] = {
            "red": 0.0, "orange_red": 0.0, "orange_yellow": 0.0,
            "brown": 0.0, "white": 0.0, "dark": 0.0,
        }

    logger.debug("Step-1 part-aware: extracted %d traits", len(visible_traits))
    return {"visible_traits": visible_traits}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract(
    image_bytes: bytes,
    part_masks: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Full Step-1 visual trait extraction.

    Args:
        image_bytes: Raw image bytes.
        part_masks: Optional dict from YoloPartMasks. If provided, segmentation
            is skipped and those masks are used directly. If None, the extractor
            may run its own YOLO segmentation internally for standalone mode.

    Returns:
        {"visible_traits": {...}}
    """
    return _part_aware_extract(image_bytes, part_masks)
