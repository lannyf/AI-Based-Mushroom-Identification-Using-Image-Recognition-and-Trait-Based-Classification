"""
Visual Trait Extractor — Step 1 of the mushroom identification pipeline.

Analyses a raw image using computer vision to produce:
  1. A colour-and-shape based species prediction (top-k list with confidence).
  2. A structured dictionary of visible traits that can be fed directly into
     the downstream LLM traversal step (Step 2).

No trained neural-network weights are required; all analysis is done with
classical CV (OpenCV + colour clustering).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

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


def analyse_colours(bgr: np.ndarray) -> Dict[str, Any]:
    """
    Extract colour profile from image.

    Returns a dict with:
      dominant_color  — string label for the most prominent colour
      secondary_color — second most prominent
      red_ratio       — fraction of pixels that are "red"
      orange_yellow_ratio
      brown_ratio
      white_ratio
      dark_ratio      — black + dark-grey pixels
    """
    small = cv2.resize(bgr, (128, 128))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    clusters = _dominant_hsv(hsv, n_clusters=5)
    colour_names = [_hsv_to_name(*c) for c in clusters]

    dominant = colour_names[0] if colour_names else "unknown"
    secondary = colour_names[1] if len(colour_names) > 1 else dominant

    # Ratio helpers
    arr = bgr.astype(np.float32)
    r, g, b = arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]

    red_ratio          = float(np.mean((r > 150) & (r > g * 1.25) & (r > b * 1.25)))
    # Orange-red: Fly Agaric caps photograph as orange-red in natural light
    # (high red, moderate green, low blue) — distinct from chanterelle orange-yellow
    orange_red_ratio   = float(np.mean((r > 140) & (g > 50) & (g < 130) & (b < 80)))
    orange_yellow_ratio = float(np.mean((r > 140) & (g > 90) & (b < 140)))
    brown_ratio        = float(np.mean((r > 80)  & (g > 45) & (b < 90) & (r > g) & (g > b)))
    white_ratio        = float(np.mean((r > 185) & (g > 185) & (b > 185)))
    dark_ratio         = float(np.mean((r < 60)  & (g < 60)  & (b < 60)))

    # Full-scene photos can let forest background dominate KMeans even when the
    # subject is a classic red Fly Agaric cap. Preserve that foreground cue in
    # the structured traits so Step 2 does not branch on irrelevant colours.
    if red_ratio >= 0.09 and white_ratio >= 0.05 and red_ratio > orange_yellow_ratio:
        dominant = "red"
        secondary = "white"

    return {
        "dominant_color":        dominant,
        "secondary_color":       secondary,
        "red_ratio":             round(red_ratio, 3),
        "orange_red_ratio":      round(orange_red_ratio, 3),
        "orange_yellow_ratio":   round(orange_yellow_ratio, 3),
        "brown_ratio":           round(brown_ratio, 3),
        "white_ratio":           round(white_ratio, 3),
        "dark_ratio":            round(dark_ratio, 3),
    }


def _looks_like_fly_agaric_signature(colour: Dict[str, Any], shape: Dict[str, Any]) -> bool:
    red = float(colour["red_ratio"])
    orange_red = float(colour.get("orange_red_ratio", 0.0))
    orange_yellow = float(colour["orange_yellow_ratio"])
    white = float(colour["white_ratio"])
    shape_name = str(shape.get("cap_shape", "")).lower()

    spotted_red_cap = red >= 0.09 and white >= 0.05 and red > orange_yellow
    warm_red_cap = (
        red >= 0.08
        and orange_red >= 0.03
        and shape_name in {"convex", "flat", "bell-shaped"}
    )
    pale_spotted_cap = white >= 0.10 and red >= 0.07

    return spotted_red_cap or warm_red_cap or pale_spotted_cap


# ---------------------------------------------------------------------------
# Shape analysis
# ---------------------------------------------------------------------------

def analyse_shape(bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimate cap shape from the image contour.

    Returns:
      cap_shape     — one of: convex, flat, funnel-shaped, bell-shaped,
                      wavy, irregular
      aspect_ratio  — width / height of bounding box (float)
      circularity   — 4π·area / perimeter² (0-1; 1 = perfect circle)
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

    # Classify shape heuristically
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

    return {
        "cap_shape":    cap_shape,
        "aspect_ratio": round(float(aspect_ratio), 2),
        "circularity":  round(float(circularity), 2),
    }


# ---------------------------------------------------------------------------
# Texture analysis
# ---------------------------------------------------------------------------

def analyse_texture(bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimate surface texture via edge density.

    Returns:
      surface_texture — 'smooth', 'fibrous', or 'scaly'
      edge_density    — fraction of edge pixels (0-1)
      has_ridges      — True if linear parallel structures detected (chanterelle cue)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # Hough line density as a proxy for ridges / gills
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=20, maxLineGap=5)
    has_ridges = lines is not None and len(lines) > 15

    if edge_density < 0.05:
        texture = "smooth"
    elif edge_density < 0.15:
        texture = "fibrous"
    else:
        texture = "scaly"

    return {
        "surface_texture": texture,
        "edge_density":    round(edge_density, 3),
        "has_ridges":      bool(has_ridges),
    }


# ---------------------------------------------------------------------------
# Brightness / exposure
# ---------------------------------------------------------------------------

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
# Species scoring from visual features
# ---------------------------------------------------------------------------

# All species handled by the image scorer (mirrors TARGET_SPECIES in api/main.py)
_ALL_SPECIES = [
    "Fly Agaric", "Chanterelle", "False Chanterelle", "Porcini",
    "Other Boletus", "Amanita virosa", "Black Trumpet",
]


def score_species(
    colour: Dict[str, Any],
    shape:  Dict[str, Any],
    texture: Dict[str, Any],
) -> Dict[str, float]:
    """
    Score each candidate species using visual features.

    Combines colour, shape, and texture signals with weighted rules.
    Returns raw (unnormalised) scores.
    """
    scores: Dict[str, float] = {sp: 0.02 for sp in _ALL_SPECIES}

    dom   = colour["dominant_color"]
    r_r   = colour["red_ratio"]
    or_r  = colour.get("orange_red_ratio", 0.0)
    oy_r  = colour["orange_yellow_ratio"]
    br_r  = colour["brown_ratio"]
    wh_r  = colour["white_ratio"]
    dk_r  = colour["dark_ratio"]
    shape_name = shape["cap_shape"]
    has_ridges = texture["has_ridges"]
    texture_name = texture["surface_texture"]

    # --- colour signals ---
    # orange_red captures the typical Fly Agaric cap colour photographed in natural light
    scores["Fly Agaric"]        += r_r * 5.0 + or_r * 4.0 + wh_r * 1.4
    scores["Amanita virosa"]    += wh_r * 2.5
    # Dampen Chanterelle when orange_red+white is present (Fly Agaric pattern)
    chanterelle_oy = max(0.0, oy_r - or_r) if wh_r > 0.02 else oy_r
    scores["Chanterelle"]       += chanterelle_oy * 4.0
    scores["False Chanterelle"] += oy_r * 2.6
    scores["Porcini"]           += br_r * 4.2
    scores["Other Boletus"]     += br_r * 3.3
    scores["Black Trumpet"]     += dk_r * 3.5 + (1.0 - wh_r) * 0.15

    if dom in {"orange", "yellow", "orange-yellow"}:
        scores["Chanterelle"]       += 0.6
        scores["False Chanterelle"] += 0.3
    if dom in {"orange", "orange-yellow", "red"} and wh_r > 0.02:
        # Orange/red cap + white elements → Fly Agaric pattern (white spots/warts)
        scores["Fly Agaric"] += 0.7
        scores["Chanterelle"] -= 0.4
    if dom == "red":
        scores["Fly Agaric"] += 0.8
    if dom == "white":
        scores["Amanita virosa"] += 0.6
    if dom in {"brown", "olive-brown", "tan"}:
        scores["Porcini"]      += 0.4
        scores["Other Boletus"] += 0.35
    if dom in {"black", "grey"}:
        scores["Black Trumpet"] += 0.7

    # --- shape signals ---
    if shape_name == "funnel-shaped":
        scores["Chanterelle"]       += 0.5
        scores["False Chanterelle"] += 0.3
        scores["Black Trumpet"]     += 0.2
    if shape_name in {"convex", "flat"}:
        scores["Porcini"]      += 0.25
        scores["Other Boletus"] += 0.2
    if shape_name == "bell-shaped":
        scores["Fly Agaric"]     += 0.35
        scores["Amanita virosa"] += 0.25
    if shape_name == "wavy":
        scores["Chanterelle"]       += 0.4
        scores["False Chanterelle"] += 0.2

    # --- texture / ridge signals ---
    if has_ridges:
        scores["Chanterelle"]       += 0.5
        scores["False Chanterelle"] += 0.25
    if texture_name == "scaly":
        scores["Fly Agaric"]     += 0.3
        scores["Amanita virosa"] += 0.1
    if texture_name == "smooth":
        scores["Chanterelle"] += 0.1
        scores["Porcini"]     += 0.1

    return scores


def _normalise(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        even = 1.0 / len(scores)
        return {k: even for k in scores}
    return {k: max(v, 0.0) / total for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyse_colours_masked(bgr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    # Compute colour stats only on mask-positive pixels. Fall back to full-image if mask too small.
    mask_bool = mask > 0
    if mask_bool.sum() < 50:
        return analyse_colours(bgr)
    pixels = bgr[mask_bool]
    # reuse existing pipeline on the masked pixels
    small = cv2.resize(pixels.reshape(-1, 3), (128, 128)) if len(pixels) >= 128 else pixels
    hsv = cv2.cvtColor(small.reshape(-1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    clusters = _dominant_hsv(hsv, n_clusters=4)
    colour_names = [_hsv_to_name(*c) for c in clusters]
    dominant = colour_names[0] if colour_names else "unknown"
    secondary = colour_names[1] if len(colour_names) > 1 else dominant
    arr = pixels.astype(np.float32)
    r, g, b = arr[:, 2], arr[:, 1], arr[:, 0]
    red_ratio = float(np.mean((r > 150) & (r > g * 1.25) & (r > b * 1.25)))
    orange_red_ratio = float(np.mean((r > 140) & (g > 50) & (g < 130) & (b < 80)))
    orange_yellow_ratio = float(np.mean((r > 140) & (g > 90) & (b < 140)))
    brown_ratio = float(np.mean((r > 80)  & (g > 45) & (b < 90) & (r > g) & (g > b)))
    white_ratio = float(np.mean((r > 185) & (g > 185) & (b > 185)))
    dark_ratio = float(np.mean((r < 60)  & (g < 60)  & (b < 60)))
    if red_ratio >= 0.09 and white_ratio >= 0.05 and red_ratio > orange_yellow_ratio:
        dominant = "red"
        secondary = "white"
    return {
        "dominant_color": dominant,
        "secondary_color": secondary,
        "red_ratio": round(red_ratio, 3),
        "orange_red_ratio": round(orange_red_ratio, 3),
        "orange_yellow_ratio": round(orange_yellow_ratio, 3),
        "brown_ratio": round(brown_ratio, 3),
        "white_ratio": round(white_ratio, 3),
        "dark_ratio": round(dark_ratio, 3),
    }


def analyse_texture_masked(bgr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    mask_bool = mask > 0
    if mask_bool.sum() < 50:
        return analyse_texture(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges[~mask_bool] = 0
    edge_density = float(np.mean(edges > 0))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=5)
    has_ridges = lines is not None and len(lines) > 10
    if edge_density < 0.05:
        texture = "smooth"
    elif edge_density < 0.15:
        texture = "fibrous"
    else:
        texture = "scaly"
    return {"surface_texture": texture, "edge_density": round(edge_density, 3), "has_ridges": bool(has_ridges)}


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
    # Use mask contour as primary source; fallback to analyse_shape when ambiguous
    mask_u = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return analyse_shape(bgr)
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


def extract(image_bytes: bytes) -> Dict[str, Any]:
    """
    Full Step-1 analysis with optional segmentation metadata and masked-trait replacement.
    """
    import io as _io
    from config import segmentation_config as seg_cfg

    pil_img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # --- CV analysis (always runs — needed for visible_traits) ---
    colour = analyse_colours(bgr)
    shape = analyse_shape(bgr)
    texture = analyse_texture(bgr)
    brightness = analyse_brightness(bgr)
    fly_agaric_like = _looks_like_fly_agaric_signature(colour, shape)

    visible_traits: Dict[str, Any] = {
        "dominant_color": colour["dominant_color"],
        "secondary_color": colour["secondary_color"],
        "cap_shape": shape["cap_shape"],
        "surface_texture": texture["surface_texture"],
        "has_ridges": texture["has_ridges"] and not fly_agaric_like,
        "brightness": brightness,
        "colour_ratios": {
            "red": colour["red_ratio"],
            "orange_red": colour.get("orange_red_ratio", 0.0),
            "orange_yellow": colour["orange_yellow_ratio"],
            "brown": colour["brown_ratio"],
            "white": colour["white_ratio"],
            "dark": colour["dark_ratio"],
        },
    }

    # --- Species scoring: CNN preferred, CV as fallback ---
    method = "cv_fallback"
    ordered: List[Tuple[str, float]] = []

    try:
        from models.cnn_classifier import get_classifier
        cnn = get_classifier()
        if cnn.is_trained:
            cnn_scores = cnn.predict(image_bytes)
            if cnn_scores is not None:
                ordered = sorted(cnn_scores.items(), key=lambda x: x[1], reverse=True)
                method = "cnn"
                logger.debug("Step-1: using CNN predictions")
    except Exception as exc:
        logger.debug("CNN unavailable, using CV fallback: %s", exc)

    if not ordered:
        raw_scores = score_species(colour, shape, texture)
        norm_scores = _normalise(raw_scores)
        ordered = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)

    top_species, top_conf = ordered[0]

    if method == "cnn":
        reasoning = (
            f"EfficientNet-B3 CNN prediction — "
            f"dominant colour '{colour['dominant_color']}', "
            f"cap shape '{shape['cap_shape']}'."
        )
    else:
        reasoning = (
            f"CV fallback (no trained CNN weights) — "
            f"dominant colour '{colour['dominant_color']}', "
            f"cap shape '{shape['cap_shape']}', "
            f"texture '{texture['surface_texture']}'"
            + (", ridges detected" if texture["has_ridges"] else "")
            + ". Train the CNN with scripts/train_cnn.py for real ML predictions."
        )

    ml_prediction: Dict[str, Any] = {
        "top_species": top_species,
        "confidence": round(top_conf, 4),
        "method": method,
        "top_k": [{"species": sp, "confidence": round(sc, 4)} for sp, sc in ordered[:5]],
        "reasoning": reasoning,
    }
    visible_traits["ml_top_species"] = top_species
    visible_traits["ml_confidence"] = round(top_conf, 4)

    # --- Segmentation metadata and optional masked trait replacement ---
    selected_mask = None
    try:
        # Only run segmentation if explicitly enabled for metadata or masked traits
        if seg_cfg.USE_MASK_FOR_TRAITS or getattr(seg_cfg, "RUN_SEGMENTATION_METADATA", False):
            from models.mushroom_segmenter import get_segmenter
            seg = get_segmenter()
            seg_res = seg.segment(image_bytes)
            instances = seg_res.get("instances", [])
            sel_idx = seg_res.get("selected_index")
            if sel_idx is not None and instances:
                sel = instances[sel_idx]
                # Only add metadata fields when RUN_SEGMENTATION_METADATA is enabled
                if getattr(seg_cfg, "RUN_SEGMENTATION_METADATA", False):
                    visible_traits["localization_source"] = "segmentation"
                    visible_traits["localization_confidence"] = round(float(sel.get("model_confidence", 0.0)), 3)
                    visible_traits["bbox"] = sel.get("bbox")
                    visible_traits["mask_used"] = False
                    visible_traits["localization_metadata"] = {
                        "foreground_area_ratio": sel.get("area_ratio"),
                        "mask_fragment_count": sel.get("fragment_count"),
                        "hole_ratio": sel.get("hole_ratio"),
                        "boundary_irregularity": sel.get("boundary_irregularity"),
                    }
                selected_mask = sel.get("cleaned_mask")
    except ImportError:
        logger.debug("Segmentation dependency not installed; skipping segmentation metadata")
    except Exception as exc:
        logger.debug("Segmentation failed: %s", exc)

    # Condition to accept masked traits
    try:
        if selected_mask is not None and seg_cfg.USE_MASK_FOR_TRAITS:
            sel_ok = (
                float(visible_traits.get("localization_confidence", 0.0)) >= seg_cfg.MIN_MASK_CONFIDENCE
                and float(visible_traits.get("localization_metadata", {}).get("foreground_area_ratio", 0.0)) >= seg_cfg.MIN_FOREGROUND_AREA_RATIO
                and float(visible_traits.get("localization_metadata", {}).get("foreground_area_ratio", 0.0)) <= seg_cfg.MAX_NEAR_FULL_FRAME_RATIO
                and int(visible_traits.get("localization_metadata", {}).get("fragment_count", 99)) <= seg_cfg.MAX_FRAGMENTATION
                and float(visible_traits.get("localization_metadata", {}).get("hole_ratio", 1.0)) <= seg_cfg.MAX_HOLE_RATIO
                and float(visible_traits.get("localization_metadata", {}).get("boundary_irregularity", 1.0)) <= seg_cfg.MAX_BOUNDARY_IRREGULARITY
            )
            if sel_ok:
                # apply cleanup conservatively
                from models.mushroom_segmenter import Segmenter as _SegClass
                # we assume cleaned_mask is already present from the segmenter
                mask_for_traits = selected_mask
                # compute masked traits
                m_colour = analyse_colours_masked(bgr, mask_for_traits)
                m_shape = analyse_shape_masked(bgr, mask_for_traits)
                m_texture = analyse_texture_masked(bgr, mask_for_traits)
                m_brightness = analyse_brightness_masked(bgr, mask_for_traits)
                # Replace trait families (policy: always use masked trait for active families)
                visible_traits["dominant_color"] = m_colour["dominant_color"]
                visible_traits["secondary_color"] = m_colour["secondary_color"]
                visible_traits["colour_ratios"]["red"] = m_colour["red_ratio"]
                visible_traits["colour_ratios"]["orange_red"] = m_colour.get("orange_red_ratio", 0.0)
                visible_traits["colour_ratios"]["orange_yellow"] = m_colour.get("orange_yellow_ratio", 0.0)
                visible_traits["colour_ratios"]["brown"] = m_colour.get("brown_ratio", 0.0)
                visible_traits["colour_ratios"]["white"] = m_colour.get("white_ratio", 0.0)
                visible_traits["colour_ratios"]["dark"] = m_colour.get("dark_ratio", 0.0)
                visible_traits["cap_shape"] = m_shape.get("cap_shape", visible_traits.get("cap_shape"))
                visible_traits["surface_texture"] = m_texture.get("surface_texture", visible_traits.get("surface_texture"))
                visible_traits["has_ridges"] = m_texture.get("has_ridges", visible_traits.get("has_ridges")) and not fly_agaric_like
                visible_traits["brightness"] = m_brightness
                visible_traits["mask_used"] = True
                # mark trait ownership
                visible_traits["trait_source_by_key"] = {k: "mask" for k in ["dominant_color", "cap_shape", "surface_texture", "has_ridges", "brightness"]}
    except Exception as exc:
        logger.debug("Error applying masked traits: %s", exc)

    logger.debug("Step-1 result: %s (%.4f) via %s", top_species, top_conf, method)
    return {"ml_prediction": ml_prediction, "visible_traits": visible_traits}
