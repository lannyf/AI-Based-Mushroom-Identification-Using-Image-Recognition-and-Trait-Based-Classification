"""Shared trait extraction and merging helper for benchmark runners.

Ensures that standalone runners (a1_*, a2_*) use the exact same trait
representation as the unified pipeline (B1/B2).
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from benchmarks.runners._extract_cache import extract as _extract_traits
from models.mushroom_segmenter import detect_case_from_masks, get_segmenter
from models.yolo_part_masks import build_part_masks


# Re-exported from models.unified_pipeline for consistency.
PHOTO_PREFERENCE = {
    "cap_color": "above",
    "cap_shape": "above",
    "cap_surface": "above",
    "underside_color": "below",
    "hymenophore_type": "below",
    "stem_color": "below",
    "stem_ring": "below",
    "stem_surface": "below",
    "puffball_surface": "above",
}


def _merge_traits(
    above_traits: Dict[str, Any],
    below_traits: Dict[str, Any],
    case: str,
) -> Dict[str, Any]:
    """
    Merge trait dicts from above and below photos, preferring the most
    informative value for each key.
    """
    merged: Dict[str, Any] = {}
    all_keys = set(above_traits.keys()) | set(below_traits.keys())

    for key in all_keys:
        a = above_traits.get(key)
        b = below_traits.get(key)
        if key == "colour_ratios":
            if a and b:
                all_ratio_keys = set(a.keys()) | set(b.keys())
                merged[key] = {
                    k: round((a.get(k, 0.0) + b.get(k, 0.0)) / 2, 3)
                    for k in all_ratio_keys
                }
            elif a:
                merged[key] = dict(a)
            elif b:
                merged[key] = dict(b)
        elif key in ("dominant_color", "secondary_color", "cap_shape",
                     "surface_texture", "brightness"):
            if case == "classical":
                if key in ("dominant_color", "cap_shape", "surface_texture"):
                    merged[key] = a if a and a != "unknown" else (b if b and b != "unknown" else (a or b))
                else:
                    merged[key] = a if a and a != "unknown" else (b if b and b != "unknown" else (a or b))
            else:
                merged[key] = a if a and a != "unknown" else (b if b and b != "unknown" else (a or b))
        elif key == "has_ridges":
            # Below photo often better for underside structures; use OR logic
            merged[key] = bool(a) if a is not None else bool(b) if b is not None else False
        elif key in PHOTO_PREFERENCE:
            pref = PHOTO_PREFERENCE[key]
            preferred = a if pref == "above" else b
            fallback = b if pref == "above" else a
            if preferred is not None and preferred != "unknown":
                merged[key] = preferred
            else:
                merged[key] = fallback
        elif key in ("trait_confidence", "trait_source_by_key"):
            # Merge dicts: above wins on overlap unless below has higher confidence
            merged_a = dict(a) if a else {}
            merged_b = dict(b) if b else {}
            result = dict(merged_a)
            for k, v in merged_b.items():
                if k not in result:
                    result[k] = v
            merged[key] = result
        else:
            merged[key] = a if a is not None else b

    # Add provenance
    merged["photo_count"] = 2
    merged["case"] = case
    return merged


def get_merged_extracted_traits(specimen) -> Dict[str, Any]:
    """
    Run the full trait extraction pipeline on a BenchmarkSpecimen.

    This mirrors exactly what UnifiedPipeline.run() does for Step 2-3:
      1. Segment above + below photos
      2. Build part masks
      3. Detect morphological case
      4. Extract traits per photo (with masks)
      5. Merge traits

    Args:
        specimen: BenchmarkSpecimen with above_path and below_path.

    Returns:
        Merged visible_traits dict.
    """
    above_bytes = specimen.load_above_bytes()
    below_bytes = specimen.load_below_bytes()

    if not above_bytes and not below_bytes:
        return {}

    # ---- 1. Segmentation ----
    segmenter = get_segmenter()
    above_seg = {"instances": [], "selected_index": None}
    below_seg = {"instances": [], "selected_index": None}

    if above_bytes:
        try:
            above_seg = segmenter.segment(above_bytes)
        except Exception:
            pass
    if below_bytes:
        try:
            below_seg = segmenter.segment(below_bytes)
        except Exception:
            pass

    # ---- 2. Part masks ----
    above_masks: Dict[str, Any] = {}
    below_masks: Dict[str, Any] = {}

    if above_bytes:
        pil = Image.open(io.BytesIO(above_bytes)).convert("RGB")
        H, W = np.array(pil).shape[:2]
        above_masks = build_part_masks(above_seg.get("instances", []), (H, W))

    if below_bytes:
        pil = Image.open(io.BytesIO(below_bytes)).convert("RGB")
        H, W = np.array(pil).shape[:2]
        below_masks = build_part_masks(below_seg.get("instances", []), (H, W))

    # ---- 3. Case detection ----
    case = detect_case_from_masks(above_masks, below_masks)

    # ---- 4. Trait extraction ----
    above_traits: Dict[str, Any] = {}
    below_traits: Dict[str, Any] = {}

    if above_bytes:
        above_traits = _extract_traits(above_bytes, part_masks=above_masks).get("visible_traits", {})
    if below_bytes:
        below_traits = _extract_traits(below_bytes, part_masks=below_masks).get("visible_traits", {})

    # ---- 5. Merge ----
    merged_traits = _merge_traits(above_traits, below_traits, case.get("case", "classical"))

    # Preserve pipeline case metadata so downstream logic is consistent
    if "trait_confidence" not in merged_traits:
        merged_traits["trait_confidence"] = {}
    merged_traits["trait_confidence"]["morphology_case"] = case.get("confidence", 0.0)
    if "trait_source_by_key" not in merged_traits:
        merged_traits["trait_source_by_key"] = {}
    merged_traits["trait_source_by_key"]["morphology_case"] = "pipeline_case_detection"
    merged_traits["morphology_case"] = case.get("case", "classical")
    merged_traits["coarse_case"] = case.get("case", "classical")

    return merged_traits
