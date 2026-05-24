> **UPDATE (2026-05-22):** The `Coral` class has been removed from the pipeline. The YOLO model is now **3-class** (Cap, Stem, Underside), and the trait extractor always runs in part-aware mode. The `ENABLE_PART_AWARE_*` feature flags have been removed. This document is kept for historical context.


# Trait Extractor Refactor Plan - May 15, 2026

## Purpose

The trait extractor originally analyzed whole images and relied on classical image processing, including Otsu thresholding, to infer mushroom shape and appearance. After introducing YOLOv8 segmentation, mushroom isolation is no longer the extractor's main responsibility. YOLO now supplies masks for mushroom parts, so the extractor should analyze the segmented mushroom regions instead of the full photo.

This refactor is needed because the current implementation is only partially mask-aware. It can use a segmentation mask, but in the unified pipeline it selects one "best" YOLO instance and extracts all traits from that single mask. If the highest-confidence instance is a stem or underside, cap traits may be computed from the wrong part. The extractor should be class-aware: cap traits should come from the cap mask, stem traits from the stem mask, underside traits from the underside mask, and coral traits from the coral mask.

The extractor must remain an evidence generator, not a final species classifier. Its output should be structured visual traits, confidence values, and provenance that downstream tree traversal, database comparison, CNN/LLM synthesis, and UI flows can use safely.

## Deployment Position

This plan is ready to implement, but deployment must be staged. The first production rollout must expose the new part-aware traits behind feature flags without immediately letting them drive `key.xml` auto-answering or database conflicts.

Deployment is considered ready only when the readiness gates in this document pass on the benchmark-image manifest.

## Current State

- `models/visual_trait_extractor.py` already contains masked variants for colour, shape, texture, and brightness analysis.
- `models/visual_trait_extractor.extract()` can run segmentation internally and applies one selected mask globally.
- `models/mushroom_segmenter.py` parses YOLOv8 segmentation results and exposes class names: `Cap`, `Coral`, `Stem`, and `Underside`.
- `models/unified_pipeline.py` calls masked extraction, but `_extract_traits_masked()` currently chooses one best instance and uses that mask for all trait extraction.
- The current unified pipeline does not call `visual_trait_extractor.extract()` inside `_extract_traits_masked()`, so the specific claim that it currently runs YOLO twice through that exact path is incorrect. The architecture issue remains: standalone `/identify`, benchmark helpers, and direct `extract()` callers would bypass part-aware logic if only `unified_pipeline.py` is changed.
- `data/raw/key.xml` contains many traits that are visually observable, but the current auto-answering in `models/key_tree_traversal.py` uses only a small subset of colour, shape, and ridge signals.
- `models/trait_database_comparator.py` currently compares only cap colour, cap shape, cap texture, ridges, and stem colour.

## Feature Flags

Add a new file `config/trait_config.py`. Do not put comparator or auto-answer flags in `segmentation_config.py`; they are not segmentation concerns. Add the flags there:

```python
ENABLE_PART_AWARE_TRAITS = False
ENABLE_PART_AWARE_KEY_AUTOANSWERS = False
ENABLE_PART_AWARE_DB_COMPARATOR = False
PART_AWARE_MIN_TRAIT_CONFIDENCE = 0.65
PART_AWARE_MIN_AUTOANSWER_CONFIDENCE = 0.80
```

Flag behavior:

- `ENABLE_PART_AWARE_TRAITS=False`: keep legacy behavior.
- `ENABLE_PART_AWARE_TRAITS=True`: compute and return new fields, but do not let them drive tree/database decisions unless the other flags are enabled.
- `ENABLE_PART_AWARE_KEY_AUTOANSWERS=True`: allow high-confidence part-aware traits to answer selected `key.xml` questions.
- `ENABLE_PART_AWARE_DB_COMPARATOR=True`: allow part-aware traits to affect database comparison.

Rollback path:

- Turn off `ENABLE_PART_AWARE_KEY_AUTOANSWERS` first if tree traversal regresses.
- Turn off `ENABLE_PART_AWARE_DB_COMPARATOR` if trait comparison creates noisy conflicts.
- Turn off `ENABLE_PART_AWARE_TRAITS` to fully return to legacy extractor output.

## Phase 0 - Baseline Capture

This phase must happen before refactor code is written.

1. Run the current extractor on every image referenced by `benchmarks/evaluation_manifest.csv`.
2. Save output to `artifacts/trait_extractor_baseline_may15.json`.
3. Record:
   - timestamp;
   - git commit hash if available;
   - manifest path;
   - manifest row count;
   - segmentation config values;
   - YOLO weights path;
   - whether segmentation was available;
   - Python package versions for `opencv`, `numpy`, `sklearn`, `ultralytics` if available.
4. Verify the manifest has 57 specimens and zero missing image paths.
5. Run the baseline extraction twice if segmentation is enabled and compare outputs. Any nondeterministic fields must be documented before refactoring.
6. Preserve the artifact. If using git for this work, commit or tag the baseline before modifying extractor code.

The baseline is required because the benchmark runner may be unavailable, but the benchmark images are available through the manifest. Improvement must be verified with before/after trait-level evidence on the same images.

## Target Architecture

There should be one shared part-aware extraction path.

### Public API

Change the public extractor entry point to accept optional precomputed part masks:

```python
def extract(
    image_bytes: bytes,
    part_masks: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Full Step-1 visual trait extraction.

    If part_masks is provided, segmentation is skipped and those masks are
    used directly. If part_masks is None, extract() may run segmentation
    internally for standalone mode when configured.
    """
```

Unified pipeline should:

1. Run YOLO once per image.
2. Normalize YOLO results into part masks.
3. Call `extract(image_bytes, part_masks=part_masks)`.
4. Merge above and below traits with explicit photo preferences.

Standalone `/identify` can continue calling `extract(image_bytes)`.

The internal standalone path must also build `YoloPartMasks` from its own segmentation result and route through the same part-specific analysers. It must not fall back to single-best-global-mask logic.

Flag routing inside `extract()`:

```python
def extract(image_bytes, part_masks=None):
    if not trait_config.ENABLE_PART_AWARE_TRAITS:
        return _legacy_extract(image_bytes)  # existing code, untouched

    if part_masks is None:
        # Standalone mode: run segmentation, build part masks, then proceed
        part_masks = _build_part_masks_from_segmentation(image_bytes)

    return _part_aware_extract(image_bytes, part_masks)
```

## YOLO Part Masks

Add a helper, either in `models/visual_trait_extractor.py` or a small dedicated module such as `models/yolo_part_masks.py`, that normalizes YOLO instances:

```python
{
    "cap": {"mask": ..., "confidence": ..., "bbox": ..., "class_name": "Cap"},
    "stem": {"mask": ..., "confidence": ..., "bbox": ..., "class_name": "Stem"},
    "underside": {"mask": ..., "confidence": ..., "bbox": ..., "class_name": "Underside"},
    "coral": {"mask": ..., "confidence": ..., "bbox": ..., "class_name": "Coral"},
    "whole": {"mask": union_of_all_parts, "confidence": ..., "bbox": ...},
}
```

Class normalization must be explicit:

```python
RAW_TO_PART_KEY = {
    "Cap": "cap",
    "Stem": "stem",
    "Underside": "underside",
    "Coral": "coral",
}
```

The helper should:

- Normalize masks to binary `uint8`.
- Remove small components per class.
- Compute mask quality per part: area ratio, fragmentation, hole ratio, boundary irregularity, confidence, and whether the mask touches image borders.
- Merge same-class masks only when they likely belong to the same specimen or tight cluster (IoU > 0.05 or within 50 px, using `np.logical_or`).
- Preserve separate-instance counts for clustered growth detection.
- Build a `whole` mask from accepted part masks.
- Fall back to the whole union mask only when the intended class mask is missing or fails quality gates.
- If no part mask passes quality gates and the union mask area ratio is below 0.01, `extract()` shall fall back to whole-image classical analysis (the current `analyse_colours`, `analyse_shape`, etc.) and set `mask_used=False`, `trait_source_by_key={}`.

Quality gate thresholds:

```python
MIN_AREA_RATIO = 0.01
MAX_FRAGMENTATION = 5
MAX_HOLE_RATIO = 0.20
MIN_CONFIDENCE = 0.30
```

## Part Routing

Extraction should route by part:

- `cap`: cap colour, cap shape, cap surface texture, cap margin, cap markings.
- `stem`: stem colour, ring, surface, thickness, bulbous base, scabers/tofsar, network pattern.
- `underside`: gills, pores, ridges/folds, teeth, underside colour, decurrent/attachment cues.
- `coral`: branching structure, branch density, branch tips, cauliflower-like folds, colour.
- `whole`: fallback colour/brightness, puffball shape, full-body morphology, quality checks.

The union mask must not be used for cap-specific traits unless the cap mask is missing or fails quality gates.

## Morphology Cases

Use two levels of morphology.

Coarse YOLO case from `mushroom_segmenter.detect_case()`:

- `coral`
- `puffball`
- `classical`
- `uncertain`

Detailed extractor case after visual analysis:

- `puffball`
- `coral`
- `classical_convex`
- `classical_concave`
- `classical_unknown`
- `uncertain`

Detailed case derivation should happen in the extractor:

```python
def derive_morphology_case(detected_parts, cap_shape, whole_shape_metrics):
    if "coral" in detected_parts:
        return "coral"
    if is_puffball_like(detected_parts, whole_shape_metrics):
        return "puffball"
    if "cap" in detected_parts and ("stem" in detected_parts or "underside" in detected_parts):
        if cap_shape in {"funnel-shaped", "depressed"}:
            return "classical_concave"
        if cap_shape in {"convex", "flat", "bell-shaped"}:
            return "classical_convex"
        return "classical_unknown"
    return "uncertain"
```

`whole_shape_metrics` contains:

```python
whole_shape_metrics = {
    "circularity": float,      # 4π·area / perimeter²
    "aspect_ratio": float,     # w / h
    "area_ratio": float,       # mask area / image area
    "solidity": float,         # area / convex hull area
}
```

```python
def is_puffball_like(detected_parts, whole_shape_metrics):
    return (
        detected_parts == {"cap"}
        and whole_shape_metrics.get("circularity", 0.0) > 0.75
        and whole_shape_metrics.get("aspect_ratio", 1.0) < 1.3
    )
```

`detect_case()` should remain class-driven. It should not depend on cap-shape heuristics. Convex vs concave belongs in the extractor after cap-shape analysis.

Puffball detection should not require a perfect two-photo pair. A single usable round or pear-shaped cap/body mask with no confident stem or underside can return `puffball` with moderate confidence.

## Proposed Output Schema

Keep existing keys for compatibility, but add richer part-aware fields:

```python
{
    "dominant_color": "...",
    "secondary_color": "...",
    "cap_shape": "...",
    "surface_texture": "...",
    "has_ridges": true,
    "brightness": "...",
    "colour_ratios": {...},

    "morphology_case": "puffball|coral|classical_convex|classical_concave|classical_unknown|uncertain",
    "coarse_case": "puffball|coral|classical|uncertain",
    "detected_parts": ["Cap", "Stem", "Underside"],
    "hymenophore_type": "gills|pores|ridges|teeth|unknown",
    "cap_color": "...",
    "stem_color": "...",
    "underside_color": "...",
    "whole_color": "...",
    "cap_surface": "smooth|scaly|warty|spiny|hairy|viscid_unknown|unknown",
    "stem_ring": "present|absent|unknown",
    "stem_surface": "smooth|fibrous|scabers|network|unknown",
    "coral_branching": "finger_like|cauliflower_like|unknown",
    "puffball_surface": "smooth|warty|spiny|unknown",
    "clustered_growth": true,
    "trait_confidence": {
        "cap_color": 0.0,
        "cap_shape": 0.0,
        "hymenophore_type": 0.0,
        "stem_ring": 0.0
    },
    "trait_source_by_key": {
        "cap_color": "yolo_cap_mask",
        "stem_color": "yolo_stem_mask",
        "hymenophore_type": "yolo_underside_mask"
    }
}
```

Compatibility rule for legacy keys:

- `dominant_color` shall equal `cap_color` when the cap mask is valid; otherwise it shall equal `whole_color`.
- `secondary_color` shall follow the same rule.
- `surface_texture` shall equal `cap_surface` when the cap mask is valid; otherwise it shall equal the texture computed from the whole mask or the legacy global analysis.
- `has_ridges` shall be derived from the `Underside` mask analysis. It is `True` when `hymenophore_type == "ridges"` or when the underside ridge detector score exceeds a conservative threshold. It is `False` for gills, pores, or teeth.

Unknown values should be explicit. Avoid returning guessed values without confidence.

`clustered_growth` is `True` when two or more valid part masks of the same class are detected with non-overlapping bboxes (IoU < 0.1) and each passes quality gates.

## Algorithm Improvements

### Shared Mask Handling

- Normalize all YOLO masks to binary `uint8`.
- Remove small components per class.
- Keep the highest-confidence instance per class, but allow multiple instances when detecting clustered growth.
- Compute mask quality per part: area ratio, fragmentation, hole ratio, boundary irregularity, confidence, and whether the mask touches image borders.
- Fall back to the whole union mask only when the intended class mask is missing or fails quality gates.

### Colour Extraction

- Compute cap colour from `Cap`, stem colour from `Stem`, underside colour from `Underside`, and whole colour from the union mask.
- Use masked pixel sampling instead of full-image resizing.
- Preserve existing colour ratios but compute them over the relevant mask area.
- Add support for colour groups needed by `key.xml`: yellow, orange, red, brown, grey-brown, black/dark-grey, white/grey-white/yellow-white, violet/lilac.
- Make sampling deterministic:

```python
if len(pixels) > 4096:
    step = max(1, len(pixels) // 4096)
    pixels = pixels[::step][:4096]
```

### Shape Extraction

- Replace Otsu-based cap shape in the YOLO path with contour analysis on the cap mask.
- Use aspect ratio, circularity, convexity, contour solidity, upper-outline curvature, and central-depression score.
- Keep Otsu-based `analyse_shape()` only as a fallback when no valid mask exists.
- Fix overlapping shape branches. The decision tree must be mutually exclusive:

```python
if circularity > 0.80 and 0.8 <= aspect_ratio <= 1.3:
    cap_shape = "convex"
elif aspect_ratio >= 1.6 and circularity < 0.6:
    cap_shape = "flat"
elif aspect_ratio <= 0.5 and circularity >= 0.70:
    cap_shape = "bell-shaped"
elif 0.5 < aspect_ratio < 0.9 and circularity < 0.60:
    cap_shape = "funnel-shaped"
elif central_depression_score > 0.6 and aspect_ratio >= 0.9:
    cap_shape = "depressed"
elif aspect_ratio >= 0.9 and circularity < 0.45:
    cap_shape = "wavy"
elif contour_complexity > 0.8 and circularity < 0.40:
    cap_shape = "irregular"
elif aspect_ratio < 0.9 and circularity >= 0.60:
    cap_shape = "bell-shaped"
else:
    cap_shape = "unknown"
```

- Classify cap shape as `convex`, `flat`, `bell-shaped`, `funnel-shaped`, `depressed`, `wavy`, `irregular`, or `unknown`.
- A depressed cap has a central indentation: high aspect ratio (wide) with a central-depression score > 0.6.
- For puffballs, compute roundness and pear-shape score from the whole mask.
- For morel/false-morel-like cases, detect honeycomb/network vs brain-like folds using high-frequency texture on the cap/whole mask.
- Return `unknown` rather than forcing `convex` when evidence is weak.

### Texture and Surface Extraction

- Fix masked edge-density computation so the denominator is mask area, not full image area:

```python
mask_area = int(np.count_nonzero(mask_bool))
edge_density = float(np.count_nonzero(edges > 0) / max(mask_area, 1))
```

- Compute surface texture per part rather than globally.
- Add cap surface categories: smooth, scaly, warty, spiny, hairy, and coarse-scaly.
- Detect concentric scales/freckles on the cap using radial texture or blob distribution.
- Detect puffball warts/spines using blob density and high-frequency protrusion patterns.

### Underside Extraction

Use the `Underside` mask as the primary source.

- Gills: long radial/parallel line density, strong linear Hough structure, fine repeated bands.
- Ridges/folds: thicker, lower-frequency branching lines; forked/decurrent folds.
- Pores: high density of small circular or elliptical blobs, low long-line score.
- Teeth: dense short spike-like texture or small hanging protrusions.
- Attachment/decurrent cue: whether underside line/ridge structures continue toward or down the stem mask.

Return `hymenophore_type` with confidence. If confidence is low, return `unknown`.

Confidence heuristic:
> `hymenophore_type` confidence = `underside_mask_quality` × `detector_score`, where `detector_score` is the highest-scoring detector among gills, pores, ridges, and teeth. If the top score is not at least 0.3 above the second-best score, confidence is halved.

### Stem Extraction

Use the `Stem` mask where available.

- Stem colour and brightness.
- Stem aspect ratio and thickness.
- Ring detection by looking for a horizontal band or high-contrast annulus around the upper stem.
- Bulbous base from lower-stem width compared with mid-stem width.
- Scabers/tofsar from dark small blob density on the stem.
- Network pattern from crossing line structures on the stem surface.

These are medium-confidence traits and should not drive hard decisions unless confidence is high.

### Coral Extraction

Use the `Coral` mask where available.

- Skeletonize the coral mask.
- Count branch endpoints and junctions.
- Estimate branch thickness and branch density.
- Classify `finger_like` vs `cauliflower_like`:
  - finger-like: separated upright branches, many endpoints, narrow branches.
  - cauliflower-like: dense flattened lobes, high boundary complexity, broad folded sheets.
- Compare tip colour against base colour for violet-tipped or yellow-branched cases.

## key.xml Auto-Answer Strategy

Auto-answering must be staged.

### Stage A - Trait Exposure Only

Deploy part-aware traits with `ENABLE_PART_AWARE_KEY_AUTOANSWERS=False`.

Goal:

- validate output shape;
- verify trait confidence;
- compare against baseline;
- ensure no downstream API breakage.

### Stage B - Conservative Auto-Answers

Enable only high-confidence root-question mappings:

- `morphology_case = coral` -> `Den är busklik med många grenar`
- `morphology_case = puffball` -> `Päronformad eller rund`
- `hymenophore_type = ridges` -> `Undersidan har åsar eller ådror`
- `hymenophore_type = gills` -> `Undersidan har skivor`
- `hymenophore_type = pores` -> `Undersidan har rör`
- `hymenophore_type = teeth` -> `Undersidan har taggar`
- morel texture detector -> `Vindlingar eller nätmönster`

Exact `key.xml` strings must be used.

Rules:

- Never map `Underside` directly to gills, pores, ridges, or teeth. YOLO only says an underside exists.
- Require `trait_confidence >= PART_AWARE_MIN_AUTOANSWER_CONFIDENCE`.
- If uncertain, return no auto-answer and let the UI ask the user.
- Track source and confidence for every auto-answer.

## Unified Pipeline Changes

Update the unified pipeline so it does not maintain a separate trait extraction implementation:

1. Run segmentation once per image through `_segment()`.
2. Convert YOLO instances to `part_masks`.
3. Call `extract(image_bytes, part_masks=part_masks)`.
4. Merge above/below traits with explicit photo preferences.

Merge preferences:

```python
PHOTO_PREFERENCE = {
    "cap_color": "above",
    "cap_shape": "above",
    "cap_surface": "above",
    "underside_color": "below",
    "hymenophore_type": "below",
    "stem_color": "below",
    "stem_ring": "below",
    "stem_surface": "below",
    "coral_branching": "above",
    "puffball_surface": "above",
}
```

For preferred traits, take the preferred photo's value when it is not `unknown` or `None`; otherwise fall back to the other photo.

Performance constraints:

- Unified path must run no more than one YOLO inference per image.
- Part-aware CV work should not add more than 250 ms per image on the local benchmark set unless justified.

## Database Comparator Strategy

Do not make database comparison part of the first deployment.

Initial comparator behavior:

- Map `cap_color`, `stem_color`, and `underside_color` to existing colour comparison where possible.
- Map `cap_surface` to existing `CAP.surface_texture`.
- Map `stem_surface` to existing `STEM.surface`.
- Map `hymenophore_type` to existing `GILLS.attachment` only as soft evidence.
- Do not add hard conflicts from low-confidence new traits.

Post-core migration:

- Extend `species_traits.xml` only for fields that cannot be represented by existing categories.
- Then update comparator weights.

The comparator should weight part-aware traits by both trait confidence and biological usefulness. Low-confidence traits should not create strong conflicts.

## Implementation Phases

### Phase 0 - Baseline

- Generate `artifacts/trait_extractor_baseline_may15.json`.
- Verify 57 manifest specimens and zero missing paths.
- Record config, weights, and package versions.
- Run baseline twice and document nondeterminism.
- Preserve the artifact before code changes.

### Phase 1 - Shared API and Part Masks

- Add `extract(image_bytes, part_masks=None)`.
- Add `YoloPartMasks` or equivalent helper.
- Normalize `Cap`, `Stem`, `Underside`, `Coral` to lowercase part keys.
- Generate per-part and whole masks.
- Fix masked texture denominator.
- Make colour sampling deterministic.
- Keep old output keys stable.
- Update unified pipeline to call `extract(..., part_masks=...)`.
- Add `ENABLE_PART_AWARE_TRAITS` flag.

### Phase 2 - Part-Specific Traits

- Add `analyse_cap_traits`.
- Add `analyse_stem_traits`.
- Add `analyse_underside_traits`.
- Add `analyse_puffball_traits`.
- Add `analyse_coral_traits`.
- Add confidence and provenance metadata.
- Use this straw-man confidence formula for all part-aware traits:

```python
def trait_confidence(mask_quality: Dict[str, float], detector_score: float) -> float:
    area_q = min(mask_quality["area_ratio"] / 0.05, 1.0)
    frag_q = max(0.0, 1.0 - (mask_quality["fragmentation"] - 1) * 0.25)
    hole_q = max(0.0, 1.0 - mask_quality["hole_ratio"] / 0.10)
    border_penalty = 0.9 if mask_quality.get("touches_border") else 1.0
    return float(
        area_q * frag_q * hole_q * border_penalty
        * mask_quality["yolo_confidence"] * detector_score
    )
```

- Fix shape heuristic overlaps.
- Return `unknown` for weak evidence.

### Phase 3 - Unified Pipeline Integration

- Replace unified pipeline trait helper with `extract(..., part_masks=...)`.
- Add merge preferences.
- Verify only one YOLO inference per image.
- Keep part-aware traits visible but do not enable auto-answering yet (`ENABLE_PART_AWARE_KEY_AUTOANSWERS=False`).

### Phase 4 - Conservative key.xml Integration

- Add `derive_key_answers`.
- Extend `KeyTreeEngine._try_auto_answer()`.
- Enable only behind `ENABLE_PART_AWARE_KEY_AUTOANSWERS`.
- Use exact `key.xml` strings and confidence thresholds.
- Roll out Stage A first, then Stage B.

### Phase 5 - Comparator Soft Integration

- Map new traits to existing fields only.
- Do not extend `species_traits.xml` until the core extractor is stable.
- Keep low-confidence traits out of hard conflict scoring.
- Enable only behind `ENABLE_PART_AWARE_DB_COMPARATOR`.

## Deployment Readiness Gates

The refactor is not deployable until these gates pass.

### Baseline Gates

- Manifest row count is 57.
- Missing image count is 0.
- Baseline artifact exists and includes metadata.
- Baseline extraction can be repeated with no unexplained nondeterministic trait changes.

### API Compatibility Gates

- Existing tests in `tests/test_visual_trait_extractor.py` pass.
- Existing API response keys remain present.
- Legacy behavior is preserved when `ENABLE_PART_AWARE_TRAITS=False`.

### Trait Quality Gates

Using manifest images:

- `coral` scenario: at least 5 of 6 specimens return `morphology_case = coral` or a documented `coral` coarse case.
- `puffball` scenario: at least 4 of 5 specimens return `morphology_case = puffball`.
- Pore-noted samples must not return high-confidence `hymenophore_type = gills`.
- Gill-noted samples must not return high-confidence `hymenophore_type = pores`.
- No high-confidence auto-answer may contradict the manifest notes.

### Auto-Answer Gates

Before enabling `ENABLE_PART_AWARE_KEY_AUTOANSWERS`:

- `false_auto_answer_rate <= baseline false_auto_answer_rate`.
- Root-question auto-answer coverage improves or remains stable.
- Every new auto-answer includes source and confidence.
- All new answer strings exactly match `key.xml`.

### Performance Gates

- Unified pipeline performs no more than one YOLO inference per image.
- Part-aware CV adds no more than 250 ms per image on average over the manifest, or the excess is documented and accepted.

### Rollback Gates

- Turning off `ENABLE_PART_AWARE_KEY_AUTOANSWERS` disables new tree-routing behavior.
- Turning off `ENABLE_PART_AWARE_DB_COMPARATOR` disables new comparator weighting.
- Turning off `ENABLE_PART_AWARE_TRAITS` returns legacy extraction behavior.

## Verification Without the Benchmark Runner

The benchmark runner is not required to verify the trait extractor. The repository contains `benchmarks/evaluation_manifest.csv`, and those rows point to images under `data/raw/Benchmark/`.

At the time this plan was written:

- The manifest has 57 specimens.
- All referenced image files exist locally.
- Scenario counts are `confusing` 22, `ood` 17, `coral` 6, `puffball` 5, `easy` 5, and `edge_case` 2.

Trait-level metrics:

- `mask_source_correctness`: cap traits came from cap masks, stem traits from stem masks, underside traits from underside masks, coral traits from coral masks.
- `morphology_case_accuracy`: expected scenario or notes agree with extracted morphology case.
- `hymenophore_accuracy`: notes containing gills, pores, folds/ridges, or teeth agree with `hymenophore_type`.
- `puffball_accuracy`: puffball images do not invent gills, pores, or stem traits.
- `coral_accuracy`: coral images expose `morphology_case = coral` and `coral_branching`.
- `cap_colour_stability`: known colour signals remain plausible.
- `tree_auto_answer_coverage`: more `key.xml` questions are answered from image traits.
- `false_auto_answer_rate`: unsupported visual guesses do not increase.
- `db_comparable_trait_count`: more useful traits are available for comparison.
- `processing_time_ms_per_image`: average extraction latency.
- `yolo_inference_count_per_image`: must be 1.0 in unified pipeline.

The refactor counts as improved only if useful trait coverage increases without increasing confident wrong auto-answers.

## Baseline Snapshot Script

Use a small script or pytest helper before changing extractor code:

```python
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.manifest import ManifestDataset
from models.visual_trait_extractor import extract

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "benchmarks" / "evaluation_manifest.csv"
OUT = PROJECT_ROOT / "artifacts" / "trait_extractor_baseline_may15.json"


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


dataset = ManifestDataset(MANIFEST)
rows = []

for sample in dataset:
    for view, loader in [
        ("above", sample.load_above_bytes),
        ("below", sample.load_below_bytes),
    ]:
        image_bytes = loader()
        if image_bytes is None:
            continue
        rows.append({
            "specimen_id": sample.specimen_id,
            "species_id": sample.species_id,
            "scenario": sample.scenario,
            "view": view,
            "notes": sample.notes,
            "visible_traits": extract(image_bytes)["visible_traits"],
        })

artifact = {
    "metadata": {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "manifest": str(MANIFEST),
        "specimens": len(dataset),
        "records": len(rows),
    },
    "records": rows,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
```

## Test Harness Sketch

Add these tests after the schema exists. Until then, mark them skipped or expected-failing.

```python
from pathlib import Path

from benchmarks.manifest import ManifestDataset
from models.visual_trait_extractor import extract

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "benchmarks" / "evaluation_manifest.csv"


def test_manifest_images_exist():
    dataset = ManifestDataset(MANIFEST)
    assert len(dataset) == 57
    for sample in dataset:
        assert sample.above_path is None or sample.above_path.exists()
        assert sample.below_path is None or sample.below_path.exists()


def test_puffball_images_route_to_puffball():
    dataset = ManifestDataset(MANIFEST)
    for sample in dataset.by_scenario("puffball"):
        img = sample.load_above_bytes()
        assert img is not None
        traits = extract(img)["visible_traits"]
        assert traits.get("morphology_case") == "puffball"
        assert traits.get("hymenophore_type") in {None, "unknown"}


def test_coral_images_route_to_coral():
    dataset = ManifestDataset(MANIFEST)
    for sample in dataset.by_scenario("coral"):
        img = sample.load_above_bytes()
        assert img is not None
        traits = extract(img)["visible_traits"]
        assert traits.get("morphology_case") == "coral"
        assert traits.get("coral_branching") in {
            "finger_like",
            "cauliflower_like",
            "unknown",
        }


def test_bolete_notes_prefer_pores_not_gills():
    dataset = ManifestDataset(MANIFEST)
    for sample in dataset:
        if "pores" not in sample.notes.lower():
            continue
        img = sample.load_below_bytes() or sample.load_above_bytes()
        assert img is not None
        traits = extract(img)["visible_traits"]
        assert not (
            traits.get("hymenophore_type") == "gills"
            and traits.get("trait_confidence", {}).get("hymenophore_type", 0.0) >= 0.80
        )
```

## Main Risks

- Overconfident auto-answering can push the key traversal down the wrong branch.
- YOLO part masks may be missing, mislabeled, or low quality.
- `Underside` is known to be the weakest YOLO class, so underside-derived traits need conservative confidence thresholds.
- Some `key.xml` questions describe non-visual traits and must remain user questions.
- More trait fields can make downstream prompts and database comparison noisier if confidence and provenance are not preserved.
- Real mushroom photos vary heavily by angle; cap-shape extraction must return `unknown` when the evidence is weak.
- The refactor must not introduce extra neural-network calls. All new analysis should be classical CV on existing masks.

## Expected Result

After the refactor, the extractor should use YOLO class labels as part-level evidence, extract traits from the correct mushroom region, report confidence and source for each important trait, and answer more of `key.xml` only when evidence is strong enough. Production rollout should proceed in stages: trait exposure first, conservative auto-answering later, and database-comparator weighting last.
