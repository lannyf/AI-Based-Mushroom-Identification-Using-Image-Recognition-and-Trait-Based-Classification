# Implementation Plan — Unify Segmentation Evidence Across Pipeline

> **Date:** 18 May 2026  
> **Goal:** Fix the architectural decoupling where `detect_case()` uses raw YOLO instances while `build_part_masks()` (and therefore trait extraction) uses filtered masks. This causes `case=coral` with `detected_parts=[]` and makes benchmark results noisy.  
> **Context:** Based on feedback in `trait_extractor_issues_may18.md` and Codex review `trait_extractor_feedback_for_kimi_may18.md`.

---

## 1. Problem Statement

Current flow in `UnifiedPipeline.run()`:

```
1. Segment above → raw instances
2. Segment below → raw instances
3. detect_case(raw_above, raw_below) → case
4. _extract_traits_masked(above, raw_above) → part masks → traits
```

**Bug:** Step 3 may see `Coral` in raw instances and return `case=coral`. Step 4 calls `build_part_masks()`, which rejects the same coral mask via geometric filter. Result: `case=coral` but `detected_parts=[]`, traits empty.

**Additional bug:** `build_part_masks()` merges same-class masks only if they overlap or are within 50 px. Non-merging clusters are silently discarded. Real coral (branching, separated clusters) loses branches.

---

## 2. Desired End State

```
1. Segment above → raw instances
2. Segment below → raw instances
3. build_part_masks(raw_above) → above_filtered_masks
4. build_part_masks(raw_below) → below_filtered_masks
5. detect_case_from_masks(above_filtered, below_filtered) → case
6. extract(above, part_masks=above_filtered) → above_traits
7. extract(below, part_masks=below_filtered) → below_traits
8. merge_traits(above_traits, below_traits, case) → merged
```

Both case detection and trait extraction use the **same filtered evidence**.

---

## 3. Changes by File

### 3.1 `models/yolo_part_masks.py` — Preserve multiple clusters per part

**Current behavior:** `build_part_masks()` keeps only the first cluster per part; others are ignored via `instance_count += 1`.

**New behavior:** Keep ALL clusters that pass quality + geometric gates. Return the union of all accepted clusters as the final mask, plus metadata about how many clusters were accepted/rejected.

**Signature change (backward-compatible):**

```python
def build_part_masks(
    instances: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict with keys: cap, stem, underside, coral, whole.
    Each value contains:
      - mask: np.ndarray (union of ALL accepted clusters for this part)
      - confidence: float (max confidence among accepted clusters)
      - bbox: tuple
      - class_name: str
      - quality: dict (quality metrics of the union mask)
      - instance_count: int (number of accepted clusters)
      - rejected_count: int (number of clusters that failed gates)
      - accepted_clusters: List[dict]  # NEW — individual cluster metadata for debugging
    """
```

**Implementation sketch:**

```python
# Inside build_part_masks, replace the merge-then-check loop with:

accepted_clusters = []
rejected_clusters = []

for item in items:
    mask = item["mask"]
    quality = _mask_quality(mask, item["confidence"])
    
    if not _passes_quality_gate(quality, part_key):
        rejected_clusters.append({...})
        continue
    
    if part_key == "coral" and not _is_coral_like(mask):
        rejected_clusters.append({...})
        continue
    
    accepted_clusters.append({
        "mask": mask,
        "confidence": item["confidence"],
        "bbox": item["bbox"],
        "quality": quality,
    })

if accepted_clusters:
    # Union all accepted masks
    union_mask = np.zeros((H, W), dtype=np.uint8)
    for c in accepted_clusters:
        union_mask = np.maximum(union_mask, c["mask"])
    
    result[part_key] = {
        "mask": union_mask,
        "confidence": round(max(c["confidence"] for c in accepted_clusters), 3),
        "bbox": _bbox_from_mask(union_mask),
        "class_name": items[0]["class_name"],
        "quality": _mask_quality(union_mask, max_conf),
        "instance_count": len(accepted_clusters),
        "rejected_count": len(rejected_clusters),
        "accepted_clusters": accepted_clusters,
    }
```

**Note:** The current merge logic (IoU > 0.05 or dist < 50) should be REMOVED. Instead, validate each cluster independently. The union happens after validation. This preserves separated coral branches.

---

### 3.2 `models/mushroom_segmenter.py` — New `detect_case_from_masks()`

**Add a new function** alongside the existing `detect_case()` (keep old function for backward compatibility):

```python
def detect_case_from_masks(
    above_masks: Dict[str, Dict[str, Any]],
    below_masks: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Detect morphological case from FILTERED part masks.
    
    Args:
        above_masks: output of build_part_masks() for above photo
        below_masks: output of build_part_masks() for below photo
    
    Returns:
        {
            "case": "classical" | "coral" | "puffball" | "uncertain",
            "confidence": float,
            "detected_parts": ["cap", "stem", ...],
            "accepted_parts": {"above": ["cap"], "below": ["stem", "underside"]},
            "reasoning": str,
        }
    """
```

**Logic:**

```python
above_parts = set(above_masks.keys()) - {"whole"}
below_parts = set(below_masks.keys()) - {"whole"}
all_parts = above_parts | below_parts

has_coral = "coral" in all_parts
has_cap = "cap" in all_parts
has_stem = "stem" in all_parts
has_underside = "underside" in all_parts

# Coral: same rules as detect_case() but using filtered parts
if has_coral:
    coral_in_both = "coral" in above_parts and "coral" in below_parts
    max_coral_conf = max(
        above_masks.get("coral", {}).get("confidence", 0.0),
        below_masks.get("coral", {}).get("confidence", 0.0),
    )
    # If only one view has coral, require high confidence
    if coral_in_both or max_coral_conf >= 0.75:
        return {
            "case": "coral",
            "confidence": round(min(0.85, max_coral_conf), 3),
            "detected_parts": sorted(all_parts),
            "accepted_parts": {
                "above": sorted(above_parts),
                "below": sorted(below_parts),
            },
            "reasoning": f"Coral detected in filtered masks: above={above_parts}, below={below_parts}",
        }

# Classical / puffball / uncertain: same as current detect_case()
# ... (copy existing logic but using has_cap/has_stem/has_underside from all_parts)
```

**Keep `detect_case()`** for external callers that still pass raw instances. Mark it `@deprecated` or add a docstring note.

---

### 3.3 `models/unified_pipeline.py` — Restructure `run()` to build masks first

**Current code (lines 280–296):**

```python
above_seg = self._segment(above_image_bytes)
below_seg = self._segment(below_image_bytes)

case = detect_case(
    above_seg.get("instances", []),
    below_seg.get("instances", []),
)

above_traits = _extract_traits_masked(
    above_image_bytes, above_seg.get("instances", [])
)
below_traits = _extract_traits_masked(
    below_image_bytes, below_seg.get("instances", [])
)
merged_traits = _merge_traits(above_traits, below_traits, case["case"])
```

**New code:**

```python
from models.yolo_part_masks import build_part_masks
from models.mushroom_segmenter import detect_case_from_masks

# 1. Segment
above_seg = self._segment(above_image_bytes)
below_seg = self._segment(below_image_bytes)

# 2. Decode image shapes once
above_H, above_W = _get_image_shape(above_image_bytes)
below_H, below_W = _get_image_shape(below_image_bytes)

# 3. Build filtered part masks once
above_masks = build_part_masks(above_seg.get("instances", []), (above_H, above_W))
below_masks = build_part_masks(below_seg.get("instances", []), (below_H, below_W))

# 4. Case detection uses filtered masks
if trait_config.ENABLE_PART_AWARE_TRAITS:
    case = detect_case_from_masks(above_masks, below_masks)
else:
    case = detect_case(
        above_seg.get("instances", []),
        below_seg.get("instances", []),
    )

# 5. Trait extraction uses the SAME masks
if trait_config.ENABLE_PART_AWARE_TRAITS:
    above_traits = _extract_traits_with_masks(above_image_bytes, above_masks)
    below_traits = _extract_traits_with_masks(below_image_bytes, below_masks)
else:
    above_traits = _extract_traits_masked(
        above_image_bytes, above_seg.get("instances", [])
    )
    below_traits = _extract_traits_masked(
        below_image_bytes, below_seg.get("instances", [])
    )

merged_traits = _merge_traits(above_traits, below_traits, case["case"])
```

**Add helper:** `_get_image_shape(image_bytes) -> Tuple[int, int]` (decode once, avoid double-decoding).

**Add helper:** `_extract_traits_with_masks(image_bytes, part_masks)` — thin wrapper that calls `extract(image_bytes, part_masks=part_masks)`.

---

### 3.4 `models/visual_trait_extractor.py` — Fallback alias fields

**Location:** Inside `_part_aware_extract()`, in the fallback branch when `not part_masks`.

**Current fallback returns:**

```python
visible_traits = {
    "dominant_color": colour["dominant_color"],
    "secondary_color": colour["secondary_color"],
    "cap_shape": shape["cap_shape"],        # actually whole-shape fallback
    "surface_texture": texture["surface_texture"],
    "has_ridges": False,
    "brightness": brightness,
    "colour_ratios": {...},
    "mask_used": False,
    "morphology_case": "uncertain",
    "coarse_case": "uncertain",
    "detected_parts": [],
}
```

**New fallback should also return:**

```python
visible_traits = {
    # ... existing fields ...
    
    # NEW — compatibility aliases so downstream consumers don't see missing values
    "cap_color": colour["dominant_color"],
    "whole_color": colour["dominant_color"],
    "stem_color": "unknown",
    "underside_color": "unknown",
    "cap_surface": texture["surface_texture"],
    "stem_surface": "unknown",
    "hymenophore_type": "unknown",
    "coral_color": "unknown",
    "coral_branching": "unknown",
    "puffball_surface": "unknown",
    "trait_confidence": {"fallback": 0.3},  # low confidence because whole-image
    "trait_source_by_key": {k: "whole_image_fallback" for k in [...]},
}
```

**Rationale:** Benchmark scripts and database comparator read `cap_color`, `stem_color`, etc. When no part masks survive, these keys are missing even though `dominant_color` has valid data. Populating aliases makes the system consistent without pretending part-level evidence exists.

---

### 3.5 `models/visual_trait_extractor.py` — Set `trait_confidence["morphology_case"]`

Codex noted that `trait_confidence["morphology_case"]` is never set, which means `ENABLE_PART_AWARE_KEY_AUTOANSWERS` (gated at 0.80) may not actually fire.

**Fix:** In `_part_aware_extract()`, after deriving `morphology_case`, set:

```python
trait_confidences["morphology_case"] = round(case_confidence, 3)
trait_sources["morphology_case"] = "detect_case_from_masks"  # or "detect_case"
```

Where `case_confidence` comes from `detect_case_from_masks()` output or is computed from the quality of the masks that support the case.

---

## 4. Testing Plan

### 4.1 Regression test — legacy mode

```bash
ENABLE_PART_AWARE_TRAITS=False .venv/bin/python -m pytest tests/
```

Ensure `ENABLE_PART_AWARE_TRAITS=False` produces identical results.

### 4.2 Coral consistency test

Run the reproduction script from `trait_extractor_issues_may18.md` (Section 6.1) and verify:

| Check | Expected |
|-------|----------|
| `RA.BO_025` | `case=coral`, `detected_parts` contains coral-related keys |
| `RA.BO_026` | `case=coral` |
| `RA.PA_028` | `case=coral` |
| `AM.VI_007` | `case=classical` (not coral) |
| `HY.PS_019` | `case=puffball` or `uncertain` (not coral) |
| `LY.PE_037` | `case≠coral` |
| `SP.CR_039` | `case≠coral` or `case=coral` with accepted coral mask in both views |

**Critical:** No specimen should show `case=coral` with empty `detected_parts`.

### 4.3 Trait field completeness test

For each of the 10 benchmark species, verify that `merged_traits` contains:
- `cap_color` (or `whole_color`) even when `detected_parts=[]`
- `trait_confidence` dict with keys for every trait

### 4.4 Build part masks cluster preservation test

Create a synthetic image with two non-overlapping coral masks (distance > 50 px). Verify `build_part_masks()` returns a coral entry with `instance_count=2` and a union mask covering both.

---

## 5. Rollback Safety

All changes are behind `ENABLE_PART_AWARE_TRAITS`:

- When `False`: `UnifiedPipeline.run()` uses old `detect_case()` and `_extract_traits_masked()` paths.
- When `True`: new `detect_case_from_masks()` and `_extract_traits_with_masks()` paths.

Default config stays `False` until benchmarks pass.

---

## 6. Estimated Scope

| File | Lines changed | Risk |
|------|--------------|------|
| `models/yolo_part_masks.py` | ~+40, refactor merge logic | Medium — affects all part masks |
| `models/mushroom_segmenter.py` | ~+35, new function | Low — additive, old function kept |
| `models/unified_pipeline.py` | ~+25, restructure run() | Medium — central pipeline path |
| `models/visual_trait_extractor.py` | ~+20, fallback aliases | Low — only fallback branch |
| Tests / benchmarks | ~+30, new assertions | Low |
| **Total** | **~150 lines** | |

---

## 7. Open Decisions (for Codex or user)

1. **Should we completely remove the old `detect_case()` after migration, or keep it indefinitely for backward compatibility?**
2. **For color fallback, should `whole_color` = `dominant_color` or should we compute a separate whole-image color?**
3. **Should `build_part_masks()` return rejected cluster metadata in production, or only in debug mode?**
4. **If both `cap` and `coral` masks are accepted (rare), which case wins?** Current logic: coral overrides classical. Keep or change?

---

## 8. Codex Review Guardrails for Implementation

These points should be addressed before or during implementation. They refine the plan above and are intended to prevent regressions while Kimi implements the changes.

### 8.1 Legacy mode must not call new mask logic

The rollback section says all changes are behind `ENABLE_PART_AWARE_TRAITS`, but the proposed `unified_pipeline.py` sketch builds `above_masks` and `below_masks` before checking the flag.

That weakens rollback safety. If `ENABLE_PART_AWARE_TRAITS=False`, legacy mode should avoid the new `build_part_masks()` path entirely.

Recommended structure:

```python
above_seg = self._segment(above_image_bytes)
below_seg = self._segment(below_image_bytes)

if trait_config.ENABLE_PART_AWARE_TRAITS:
    above_H, above_W = _get_image_shape(above_image_bytes)
    below_H, below_W = _get_image_shape(below_image_bytes)
    above_masks = build_part_masks(above_seg.get("instances", []), (above_H, above_W))
    below_masks = build_part_masks(below_seg.get("instances", []), (below_H, below_W))
    case = detect_case_from_masks(above_masks, below_masks)
    above_traits = _extract_traits_with_masks(above_image_bytes, above_masks)
    below_traits = _extract_traits_with_masks(below_image_bytes, below_masks)
else:
    case = detect_case(above_seg.get("instances", []), below_seg.get("instances", []))
    above_traits = _extract_traits_masked(above_image_bytes, above_seg.get("instances", []))
    below_traits = _extract_traits_masked(below_image_bytes, below_seg.get("instances", []))
```

### 8.2 Test command must match config behavior

The testing plan uses:

```bash
ENABLE_PART_AWARE_TRAITS=False .venv/bin/python -m pytest tests/
```

This will only work if `config/trait_config.py` reads from the environment. Currently it defines a Python constant:

```python
ENABLE_PART_AWARE_TRAITS = False
```

Either update `trait_config.py` to read env vars, or change the test instructions to patch/mutate the Python flag during tests.

### 8.3 Preserve the classical-structure contradiction check

The proposed `detect_case_from_masks()` sketch allows any single-view coral mask with confidence >= 0.75 to return `case=coral`.

That drops an important rule from the current `detect_case()` implementation: single-view coral should not override accepted classical evidence.

Required rule:

```python
has_classical_structure = (
    (has_cap and (has_stem or has_underside))
    or (has_stem and has_underside)
)

coral_single_view = (
    (has_coral_above != has_coral_below)
    and max_coral_conf >= 0.75
    and not has_classical_structure
)
```

Keep this rule in `detect_case_from_masks()`. Otherwise accepted cap/stem/underside masks can still be routed as coral because of one high-confidence coral detection.

### 8.4 Preserve multiple clusters carefully

The plan says to keep all accepted clusters for every part. That may be correct for coral, but it is riskier for `cap`, `stem`, and `underside` because multiple accepted clusters may represent:

- several mushrooms in the photo;
- background detections;
- duplicate but spatially separate false positives.

Recommended conservative implementation:

- For `coral`: validate clusters independently and union all accepted coral clusters.
- For `cap`, `stem`, and `underside`: keep the current tight-cluster behavior unless benchmark evidence shows it is causing missed traits.

This keeps the coral recall fix focused without increasing false merges for classical mushrooms.

### 8.5 Avoid serializing raw masks inside debug metadata

The proposed `accepted_clusters` field includes individual cluster metadata. That is useful, but it should not expose raw NumPy masks in API-facing results.

Recommended shape:

```python
"accepted_clusters": [
    {
        "confidence": 0.81,
        "bbox": (x, y, w, h),
        "quality": {...},
    }
]
```

Keep raw cluster masks internal while building the union mask. This avoids JSON serialization issues and large response payloads.

### 8.6 Clarify who owns `trait_confidence["morphology_case"]`

The plan suggests setting `trait_confidence["morphology_case"]` inside `_part_aware_extract()`, based on `detect_case_from_masks()`.

But `visual_trait_extractor.extract()` does not receive the pipeline-level case result. It only receives per-image part masks. The final case is derived across both above and below views in `UnifiedPipeline`.

Recommended approach:

- Let `detect_case_from_masks()` return the authoritative pipeline case and confidence.
- After `_merge_traits()`, set:

```python
merged_traits["morphology_case"] = case["case"]
merged_traits.setdefault("trait_confidence", {})["morphology_case"] = case["confidence"]
merged_traits.setdefault("trait_source_by_key", {})["morphology_case"] = "detect_case_from_masks"
```

This keeps two-view case confidence in the pipeline, where both views are available.

### 8.7 Expand the regression specimen list

The coral consistency test should include all known problem cases, not only a subset.

Add these non-coral false-positive checks:

- `LY.PE_037`
- `LY.PE_038`
- `SP.CR_039`
- `SP.CR_040`

Add all real coral recall checks:

- `RA.BO_025`
- `RA.BO_026`
- `RA.BO_027`
- `RA.PA_028`
- `RA.PA_029`
- `RA.PA_030`

The key invariant remains:

```text
No specimen should have case=coral with detected_parts=[].
```

But also track the precision/recall tradeoff explicitly:

- real coral specimens should remain `case=coral`;
- known non-coral false positives should not become `case=coral` unless accepted mask evidence clearly explains why.
