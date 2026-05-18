# Trait Extractor Refactor Plan - Deployment Ready - May 15, 2026

## Purpose

The trait extractor originally analyzed whole images and used classical image processing, including Otsu thresholding, to isolate and describe a mushroom. With YOLOv8 segmentation now available, isolation should be delegated to YOLO. The trait extractor should instead measure visual traits from the correct segmented mushroom part.

The current implementation is only partially mask-aware. It can use a segmentation mask, but it mostly treats segmentation as one foreground region. If the selected YOLO instance is a stem or underside, cap colour, cap shape, and surface texture can be computed from the wrong region. The refactor is needed so:

- cap traits come from cap masks;
- stem traits come from stem masks;
- underside traits come from underside masks;
- coral traits come from coral masks;
- whole-body traits are used only for fallback, puffballs, and global morphology.

The extractor must remain an evidence generator, not a final species classifier. It should output structured visual traits, confidence values, and provenance that downstream tree traversal, database comparison, CNN/LLM synthesis, and UI flows can use safely.

## Deployment Position

This plan is ready to implement, but deployment must be staged. The first production rollout must expose the new part-aware traits behind feature flags without immediately letting them drive `key.xml` auto-answering or database conflicts.

Deployment is considered ready only when the readiness gates in this document pass on the benchmark-image manifest.

Current assessment:

- The plan is ready for implementation and staged rollout.
- The refactor itself is not ready for production deployment until the readiness gates pass.
- Production enablement should be treated as a separate decision after baseline capture, implementation, verification, and rollback checks.

## Current Code Facts

- `models/visual_trait_extractor.py` contains whole-image and masked variants for colour, shape, texture, and brightness.
- `models/visual_trait_extractor.extract()` can run segmentation internally and applies one selected mask globally.
- `models/mushroom_segmenter.py` parses YOLOv8 segmentation and exposes `Cap`, `Coral`, `Stem`, and `Underside`.
- `models/unified_pipeline.py` has its own `_extract_traits_masked()` helper that selects one best instance and applies one mask globally.
- The current unified pipeline does not call `visual_trait_extractor.extract()` inside `_extract_traits_masked()`, so the specific claim that it currently runs YOLO twice through that path is incorrect.
- The architecture issue remains: standalone `/identify`, benchmark helpers, and direct `extract()` callers would bypass part-aware logic if only `unified_pipeline.py` is changed.
- `data/raw/key.xml` contains visually observable and non-visual questions.
- `models/trait_database_comparator.py` currently compares cap colour, cap shape, cap texture, ridges, and stem colour.

## Feature Flags

Add flags in `config/segmentation_config.py` or a new trait config module:

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

## Phase 0 - Mandatory Baseline

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

The baseline is required because the benchmark runner may be unavailable, but the benchmark images are available through the manifest.

## Shared Public API

Make `models.visual_trait_extractor.extract()` the only public trait extraction path.

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
2. Normalize YOLO instances into `part_masks`.
3. Call `extract(image_bytes, part_masks=part_masks)`.
4. Merge above and below traits using explicit preferences.

Standalone `/identify` can continue calling `extract(image_bytes)`.

## YOLO Part Masks

Add a helper, preferably `models/yolo_part_masks.py`, that converts YOLO instances to normalized masks:

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

- normalize masks to binary `uint8`;
- compute area ratio, fragmentation, hole ratio, boundary irregularity, border contact, and YOLO confidence per part;
- merge same-class masks only when they likely belong to one specimen or tight cluster;
- preserve separate-instance counts for clustered growth;
- build a `whole` mask from accepted part masks.

## Part-Aware Trait Routing

Route extraction by part:

- `cap`: cap colour, cap shape, cap surface texture, cap margin, cap markings.
- `stem`: stem colour, ring, surface, thickness, bulbous base, scabers/tofsar, network pattern.
- `underside`: gills, pores, ridges/folds, teeth, underside colour, decurrent/attachment cues.
- `coral`: branching structure, branch density, branch tips, cauliflower-like folds, colour.
- `whole`: fallback colour/brightness, puffball shape, global morphology, quality checks.

Do not use the whole mask for cap-specific traits unless the cap mask is missing or fails quality gates.

## Morphology Cases

Keep two levels of morphology.

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

`detect_case()` should remain class-driven. It should not depend on cap-shape heuristics. Convex vs concave belongs in the extractor after cap-shape analysis.

Puffball detection should not require a perfect two-photo pair. A single usable round or pear-shaped body mask with no confident stem or underside can return `puffball` with moderate confidence.

## Output Schema

Keep current compatibility fields:

```python
{
    "dominant_color": "...",
    "secondary_color": "...",
    "cap_shape": "...",
    "surface_texture": "...",
    "has_ridges": true,
    "brightness": "...",
    "colour_ratios": {...},
}
```

Add part-aware fields:

```python
{
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

Unknown values should be explicit. Avoid returning guessed values without confidence.

## Algorithm Requirements

### Deterministic Colour

- Compute cap colour from `cap`, stem colour from `stem`, underside colour from `underside`, and whole colour from `whole`.
- Replace random sampling with deterministic sampling:

```python
if len(pixels) > 4096:
    step = max(1, len(pixels) // 4096)
    pixels = pixels[::step][:4096]
```

- Preserve existing colour ratios, but compute them over the relevant mask.

### Masked Texture Fix

Fix masked edge density:

```python
mask_area = int(np.count_nonzero(mask_bool))
edge_density = float(np.count_nonzero(edges > 0) / max(mask_area, 1))
```

### Shape

- Use cap-mask contours for cap shape.
- Keep Otsu-based `analyse_shape()` only as fallback.
- Fix overlapping shape rules, especially bell-shaped vs funnel-shaped.
- Return `unknown` rather than forcing `convex` when evidence is weak.

### Underside

Use the underside mask as primary source:

- gills: long radial or parallel line density;
- ridges/folds: thicker, lower-frequency branching structures;
- pores: small circular/elliptical blob density;
- teeth: short spike-like texture or hanging protrusions.

Return `hymenophore_type` only when confidence passes threshold; otherwise return `unknown`.

### Stem

Use the stem mask for:

- stem colour;
- ring detection;
- bulbous base;
- scabers/tofsar;
- network pattern.

These are medium-confidence traits and should not drive hard decisions unless confidence is high.

### Coral

Use the coral mask for:

- skeleton branch endpoints and junctions;
- branch density and thickness;
- `finger_like` vs `cauliflower_like`;
- branch tip colour vs base colour.

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

- Never map `Underside` directly to gills, pores, ridges, or teeth.
- Require `trait_confidence >= PART_AWARE_MIN_AUTOANSWER_CONFIDENCE`.
- If uncertain, return no auto-answer.
- Track source and confidence for every auto-answer.

## Database Comparator Strategy

Do not make database comparison part of the first deployment.

Initial comparator behavior:

- map `cap_color`, `stem_color`, and `underside_color` to existing colour comparison where possible;
- map `cap_surface` to existing `CAP.surface_texture`;
- map `stem_surface` to existing `STEM.surface`;
- map `hymenophore_type` to existing `GILLS.attachment` only as soft evidence;
- do not add hard conflicts from low-confidence new traits.

Post-core migration:

- extend `species_traits.xml` only for fields that cannot be represented by existing categories;
- then update comparator weights.

## Unified Pipeline Changes

Replace `_extract_traits_masked()` with the shared extractor path:

1. `_segment(image_bytes)` returns YOLO instances.
2. `YoloPartMasks.from_instances(instances, image_shape)` normalizes masks.
3. `extract(image_bytes, part_masks=part_masks)` computes traits.
4. `_merge_traits()` merges above/below outputs.

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

Performance constraint:

- unified path must run no more than one YOLO inference per image;
- part-aware CV work should not add more than 250 ms per image on the local benchmark set unless justified.

## Implementation Phases

### Phase 0 - Baseline

- Generate `artifacts/trait_extractor_baseline_may15.json`.
- Verify 57 manifest specimens and zero missing paths.
- Record config and weights.
- Run baseline twice and document nondeterminism.

### Phase 1 - Shared API and Part Masks

- Add `extract(image_bytes, part_masks=None)`.
- Add `YoloPartMasks`.
- Normalize class names.
- Generate part masks and whole mask.
- Fix masked texture denominator.
- Make colour sampling deterministic.
- Keep legacy keys stable.
- Add `ENABLE_PART_AWARE_TRAITS`.

### Phase 2 - Part-Specific Traits

- Add cap, stem, underside, puffball, and coral analysis functions.
- Add confidence and provenance metadata.
- Fix shape heuristic overlaps.
- Return `unknown` for weak evidence.

### Phase 3 - Unified Pipeline Integration

- Replace unified pipeline trait helper with `extract(..., part_masks=...)`.
- Add merge preferences.
- Verify only one YOLO inference per image.
- Keep part-aware traits visible but do not enable auto-answering yet.

### Phase 4 - Conservative key.xml Integration

- Add `derive_key_answers`.
- Extend `KeyTreeEngine._try_auto_answer()`.
- Enable only behind `ENABLE_PART_AWARE_KEY_AUTOANSWERS`.
- Use exact `key.xml` strings and confidence thresholds.

### Phase 5 - Comparator Soft Integration

- Map new traits to existing fields only.
- Do not extend `species_traits.xml` until the core extractor is stable.
- Keep low-confidence traits out of hard conflict scoring.

## Verification Without Benchmark Runner

Use `benchmarks/evaluation_manifest.csv` directly.

Known manifest facts:

- 57 specimens.
- zero missing referenced images.
- scenarios: `confusing` 22, `ood` 17, `coral` 6, `puffball` 5, `easy` 5, `edge_case` 2.

Trait-level metrics:

- `mask_source_correctness`
- `morphology_case_accuracy`
- `hymenophore_accuracy`
- `puffball_accuracy`
- `coral_accuracy`
- `cap_colour_stability`
- `tree_auto_answer_coverage`
- `false_auto_answer_rate`
- `db_comparable_trait_count`
- `processing_time_ms_per_image`
- `yolo_inference_count_per_image`

Implementation should include a small metrics script that computes these gates automatically from:

- the baseline artifact, for example `artifacts/trait_extractor_baseline_may15.json`;
- the new extractor output on the same manifest;
- the manifest notes and scenario labels.

This script is not optional for deployment. Without it, the readiness gates would be manual and error-prone.

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
- root-question auto-answer coverage improves or remains stable.
- every new auto-answer includes source and confidence.
- all new answer strings exactly match `key.xml`.

### Performance Gates

- unified pipeline performs no more than one YOLO inference per image.
- part-aware CV adds no more than 250 ms per image on average over the manifest, or the excess is documented and accepted.

### Rollback Gates

- Turning off `ENABLE_PART_AWARE_KEY_AUTOANSWERS` disables new tree-routing behavior.
- Turning off `ENABLE_PART_AWARE_DB_COMPARATOR` disables new comparator weighting.
- Turning off `ENABLE_PART_AWARE_TRAITS` returns legacy extraction behavior.

## Baseline Snapshot Script

Use a script or pytest helper before changing extractor code:

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

- Overconfident auto-answering can send key traversal down the wrong branch.
- YOLO part masks may be missing, mislabeled, or low quality.
- `Underside` is known to be the weakest YOLO class.
- Non-visual `key.xml` questions must remain user questions.
- Extra trait fields can make downstream prompts and comparator output noisy if confidence and provenance are ignored.
- Real mushroom photos vary heavily by angle; cap-shape extraction must return `unknown` when the evidence is weak.

## Expected Result

After the refactor, the extractor should use YOLO class labels as part-level evidence, extract traits from the correct mushroom region, report confidence and source for each important trait, and answer more of `key.xml` only when evidence is strong enough. Production rollout should proceed in stages: trait exposure first, conservative auto-answering later, and database-comparator weighting last.
