"""Shared trait-extraction cache for benchmark runners.

Visual trait extraction (YOLO + OpenCV) is expensive and is invoked
by multiple runners (tree, trait_db, LLM, multimodal). This module
caches results keyed by image SHA-256 so each image is analysed only
once per benchmark run.
"""

import hashlib
from typing import Any, Dict, Optional

from models.visual_trait_extractor import extract as _original_extract

# In-memory cache: sha256_hex[+mask_hash] -> extraction_result dict.
_extract_cache: dict = {}


def _hash_masks(part_masks: Optional[Dict[str, Any]]) -> str:
    """Produce a stable hash of part_masks for cache keying."""
    if part_masks is None:
        return ""
    # Simple deterministic serialization
    import json
    try:
        serialized = json.dumps(part_masks, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    except (TypeError, ValueError):
        return "mask_err"


def extract(image_bytes: bytes, part_masks: Optional[Dict[str, Any]] = None) -> dict:
    """Return cached visual traits or extract them if not yet cached.

    Args:
        image_bytes: Raw JPEG/PNG bytes.
        part_masks: Optional dict from YoloPartMasks. If provided, segmentation
            is skipped and those masks are used directly.

    Returns:
        Dictionary with keys such as ``visible_traits`` and ``bounding_boxes``.
    """
    h = hashlib.sha256(image_bytes).hexdigest()
    mask_suffix = _hash_masks(part_masks)
    cache_key = f"{h}:{mask_suffix}" if mask_suffix else h

    if cache_key not in _extract_cache:
        _extract_cache[cache_key] = _original_extract(image_bytes, part_masks=part_masks)
    return _extract_cache[cache_key]
