# Trait Extractor Refactor Plan - Final Rewrite - May 15, 2026

> Superseded: the deployment-ready rewrite is in
> `Docs/trait_extractor_refactor_may15_deployment_ready.md`.
> This draft is kept for historical context.

## Purpose

The trait extractor originally analyzed whole images and used classical image processing, including Otsu thresholding, to isolate and describe a mushroom. With YOLOv8 segmentation now available, isolation should be delegated to YOLO and the trait extractor should focus on measuring traits from the correct segmented mushroom part.

The current implementation is only partially mask-aware. It can use a mask, but it treats segmentation mostly as one foreground region. If a stem or underside is selected as the best YOLO instance, cap colour, cap shape, and texture can be computed from the wrong region. The refactor is needed so cap traits come from cap masks, stem traits from stem masks, underside traits from underside masks, and coral traits from coral masks.

The extractor must remain an evidence generator, not a final species classifier. Its output should be structured visual traits, confidence values, and provenance that downstream tree traversal, database comparison, CNN/LLM synthesis, and UI flows can use safely.

## Feedback Incorporated

The feedback on the earlier plan was mostly correct. This rewrite incorporates it with two corrections.

Valid feedback incorporated:

- Capture a mandatory baseline before changing code.
- Make `models.visual_trait_extractor.extract()` the shared public trait extraction path.
- Let `extract()` accept precomputed YOLO part masks so the unified pipeline, standalone API, tests, and benchmark helpers use the same trait logic.
- Normalize YOLO class names explicitly because YOLO returns `Cap`, `Stem`, `Underside`, and `Coral`.
- Add deterministic colour sampling so before/after comparisons are reproducible.
- Fix masked edge density so it divides by mask area, not full image area.
- Add explicit above/below merge preferences for cap, underside, stem, puffball, and coral traits.
- Use exact `key.xml` answer strings when generating auto-answers.
- Do not extend database comparator weights for fields that are not present or mappable in `species_traits.xml`.

Corrections to the feedback:

- The current `UnifiedPipeline` does not call `visual_trait_extractor.extract()` from `_extract_traits_masked()`, so it is not currently doing YOLO twice through that exact path. The architecture issue is still real: standalone `/identify`, benchmark helpers, and direct `extract()` callers would bypass a refactor done only inside `unified_pipeline.py`.
- Keep `mushroom_segmenter.detect_case()` as a coarse YOLO case router: `coral`, `puffball`, `classical`, `uncertain`. Derive `classical_convex` vs `classical_concave` inside the trait extractor after cap-shape analysis.

## Current State

- `models/visual_trait_extractor.py` contains whole-image and masked variants for colour, shape, texture, and brightness.
- `models/mushroom_segmenter.py` parses YOLOv8 segmentation and exposes `Cap`, `Coral`, `Stem`, and `Underside`.
- `models/unified_pipeline.py` has its own `_extract_traits_masked()` helper that selects one best instance and applies one mask globally.
- `models/visual_trait_extractor.extract()` also has an internal segmentation path and applies one selected mask globally when mask use is enabled.
- `data/raw/key.xml` contains many visually observable traits, but current auto-answering uses only a small subset of colour, shape, and ridge signals.
- `models/trait_database_comparator.py` currently compares only cap colour, cap shape, cap texture, ridges, and stem colour.

## Phase 0 - Baseline Capture

This phase must happen before refactor code is written.

1. Run the current extractor on every image referenced by `benchmarks/evaluation_manifest.csv`.
2. Save output to `artifacts/trait_extractor_baseline_may15.json`.
3. Include timestamp, git commit hash if available, manifest path, row count, and extractor configuration.
4. Verify that the manifest contains 57 specimens and that all image paths resolve.
5. Preserve the artifact. If using git for this work, commit or tag the baseline before modifying extractor code.

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
    internally for standalone mode.
    """
```

Unified pipeline should:

1. Run YOLO once per image.
2. Normalize YOLO results into part masks.
3. Call `extract(image_bytes, part_masks=part_masks)`.
4. Merge above and below traits with explicit photo preferences.

Standalone `/identify` can continue calling `extract(image_bytes)`.

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
- Compute area ratio, fragmentation, hole ratio, boundary irregularity, border contact, and YOLO confidence per part.
- Merge same-class masks only when they likely belong to the same specimen or tight cluster.
- Preserve separate-instance counts for clustered growth.
- Build a `whole` mask from accepted part masks.

## Part Routing

Trait extraction should route by part:

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

Puffball detection should not require a perfect two-photo pair. A single usable round or pear-shaped cap/body mask with no confident stem or underside can return `puffball` with moderate confidence.

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

## Algorithm Changes

### Colour

- Compute cap colour from `cap`, stem colour from `stem`, underside colour from `underside`, and whole colour from `whole`.
- Use deterministic masked sampling instead of `np.random.choice`.
- Preserve existing colour ratios, but compute them over the relevant mask.
- Expand colour grouping for `key.xml`: yellow, orange, red, brown, grey-brown, black/dark-grey, white/grey-white/yellow-white, violet/lilac.

Deterministic sampling:

```python
if len(pixels) > 4096:
    step = max(1, len(pixels) // 4096)
    pixels = pixels[::step][:4096]
```

### Shape

- Replace Otsu-based cap shape in the YOLO path with contour analysis on the cap mask.
- Use aspect ratio, circularity, solidity, convexity, contour irregularity, upper-outline curvature, and central-depression score.
- Keep Otsu-based `analyse_shape()` only as a fallback when no valid mask exists.
- Fix overlapping shape branches, especially bell-shaped vs funnel-shaped.

Shape labels:

- `convex`
- `flat`
- `bell-shaped`
- `funnel-shaped`
- `depressed`
- `wavy`
- `irregular`
- `unknown`

### Texture

Fix masked edge density:

```python
mask_area = int(np.count_nonzero(mask_bool))
edge_density = float(np.count_nonzero(edges > 0) / max(mask_area, 1))
```

Texture should be computed per part:

- cap surface: smooth, scaly, warty, spiny, hairy, coarse-scaly.
- stem surface: smooth, fibrous, scabers/tofsar, network.
- puffball surface: smooth, warty, spiny.

### Underside

Use the underside mask as the primary source:

- Gills: long radial or parallel line density; strong linear Hough structure; fine repeated bands.
- Ridges/folds: thicker, lower-frequency branching lines; forked/decurrent folds.
- Pores: high density of small circular or elliptical blobs; low long-line score.
- Teeth: dense short spike-like texture or small hanging protrusions.
- Decurrent cue: underside structures continue toward or down the stem mask.

Return `hymenophore_type` with confidence. If confidence is low, return `unknown`.

### Stem

Use the stem mask where available:

- Stem colour.
- Stem aspect ratio and thickness.
- Ring detection from a horizontal band or annulus-like contrast on the upper stem.
- Bulbous base from lower-stem width compared with mid-stem width.
- Scabers/tofsar from small dark blob density.
- Network pattern from crossing line structures.

### Coral

Use the coral mask where available:

- Skeletonize the coral mask.
- Count branch endpoints and junctions.
- Estimate branch density and branch thickness.
- Classify `finger_like` vs `cauliflower_like`.
- Compare tip colour against base colour for violet-tipped or yellow-branched cases.

## key.xml Auto-Answer Mapping

Add `derive_key_answers(visible_traits)` and let `KeyTreeEngine._try_auto_answer()` use those structured answers before falling back to older heuristics.

Use exact answer strings from `data/raw/key.xml`:

- `Undersidan har åsar eller ådror`
- `Undersidan har skivor`
- `Undersidan har taggar`
- `Den är busklik med många grenar`
- `Päronformad eller rund`
- `Undersidan har rör`
- `Vindlingar eller nätmönster`

Suggested mappings:

- `morphology_case = coral` -> `Den är busklik med många grenar`
- `morphology_case = puffball` -> `Päronformad eller rund`
- `hymenophore_type = ridges` -> `Undersidan har åsar eller ådror`
- `hymenophore_type = gills` -> `Undersidan har skivor`
- `hymenophore_type = pores` -> `Undersidan har rör`
- `hymenophore_type = teeth` -> `Undersidan har taggar`
- morel texture detector -> `Vindlingar eller nätmönster`

Rules:

- Never map `Underside` directly to gills, pores, ridges, or teeth. YOLO only says an underside exists.
- Use conservative thresholds.
- If uncertain, return no auto-answer and let the UI ask the user.

## key.xml Trait Coverage

High-confidence image-derived traits:

- Overall morphology: ridges, gills, teeth, coral-like branching, puffball/round, pores, morel-like folds.
- Whole mushroom colour: yellow, grey-brown, black/dark-grey, orange stem/underside.
- Cap colour: yellow, yellow-orange, wine-red, chocolate-brown/violet, chestnut/light-brown, brown, orange-brown.
- Cap shape: bell-shaped, rounded/convex, flat, funnel/depressed, irregular/wavy.
- Cap surface: smooth, warty, spiny, scaly, coarse-scaly, hairy edge.
- Coral branch appearance: finger-like, cauliflower-like, violet tips, yellow branches.
- Puffball appearance: white, grey-white, yellow-brown, warty, spiny, smooth, large round.
- Bolete/pore traits: pores vs gills, pore colour, cap colour, ring presence, stem network/scabers.
- Morel-like traits: brain-like folds vs honeycomb/networked pointed cap.

Medium-confidence image-derived traits:

- Gill attachment: decurrent vs not decurrent.
- Forked/branched gills.
- Stem ring presence.
- Stem base bulbousness.
- Stem network pattern.
- Stem scabers/tofsar colour.
- Clustered vs solitary growth when multiple fruit bodies are visible.
- Substrate hint such as wood/stump only if a scene-context classifier is added.

Traits that should usually remain user-provided:

- Smell.
- Taste.
- Latex/saft when broken.
- Flesh colour change after breaking.
- Tough vs brittle stem when physically broken.
- Tube detachability from the cap.
- Exact size in centimeters without a scale reference.
- Season, unless image metadata or user-provided date is available.
- Habitat/tree association unless the app asks the user or adds a separate habitat classifier.

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

## Database Comparator Changes

Do not blindly add comparator weights for fields that do not exist in `species_traits.xml`.

First map new visible traits to existing database fields where possible:

- `hymenophore_type = ridges|gills|pores|teeth` can inform `GILLS.attachment`.
- `cap_surface` can inform `CAP.surface_texture`.
- `stem_surface` can inform `STEM.surface`.
- `stem_color`, `cap_color`, and `underside_color` can reuse existing colour comparison logic.

Only after that, extend `species_traits.xml` for genuinely new traits:

- `morphology_case`
- `stem_ring`
- `puffball_surface`
- `coral_branching`
- `clustered_growth`

The comparator should weight part-aware traits by both trait confidence and biological usefulness. Low-confidence traits should not create strong conflicts.

## Implementation Phases

### Phase 0 - Baseline

- Run the baseline snapshot script.
- Save `artifacts/trait_extractor_baseline_may15.json`.
- Verify all 57 manifest specimens resolve.
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

### Phase 2 - Part-Specific Traits

- Add `analyse_cap_traits`.
- Add `analyse_stem_traits`.
- Add `analyse_underside_traits`.
- Add `analyse_puffball_traits`.
- Add `analyse_coral_traits`.
- Add confidence and provenance metadata.
- Fix shape heuristic overlaps.

### Phase 3 - key.xml Integration

- Add `derive_key_answers`.
- Extend `KeyTreeEngine._try_auto_answer()`.
- Use exact Swedish `key.xml` strings.
- Add conservative confidence thresholds.
- Track auto-answer source and confidence.

### Phase 4 - Database Comparator

- First map new visible traits to existing `species_traits.xml` fields.
- Then extend `species_traits.xml` only where needed.
- Update `_compare_visible_to_db` and `_TRAIT_WEIGHTS` after data support exists.

### Phase 5 - Tests and Verification

- Add synthetic part-mask tests.
- Add real-image tests using `benchmarks/evaluation_manifest.csv`.
- Compare new output against `artifacts/trait_extractor_baseline_may15.json`.
- Re-run the full benchmark later when the benchmark runner is available.

## Verification Without the Benchmark Runner

The benchmark runner is not required to verify the trait extractor. The repository contains `benchmarks/evaluation_manifest.csv`, and those rows point to images under `data/raw/Benchmark/`.

At the time this plan was written:

- The manifest has 57 specimens.
- All referenced image files exist locally.
- Scenario counts are `confusing` 22, `ood` 17, `coral` 6, `puffball` 5, `easy` 5, and `edge_case` 2.

Trait-level metrics:

- `mask_source_correctness`: cap traits come from cap masks, stem traits from stem masks, underside traits from underside masks, coral traits from coral masks.
- `morphology_case_accuracy`: scenario/notes agree with extracted morphology case.
- `hymenophore_accuracy`: notes containing gills, pores, folds/ridges, or teeth agree with `hymenophore_type`.
- `puffball_accuracy`: puffball images do not invent gills, pores, or stem traits.
- `coral_accuracy`: coral images expose `morphology_case = coral` and `coral_branching`.
- `cap_colour_stability`: known colour signals remain plausible.
- `tree_auto_answer_coverage`: more `key.xml` questions are answered from image traits.
- `false_auto_answer_rate`: unsupported visual guesses do not increase.
- `db_comparable_trait_count`: more useful traits are available for comparison.

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

Add tests after the new schema exists:

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
        assert traits.get("hymenophore_type") != "gills"
```

These tests should be skipped or marked expected-failing until the new output schema is implemented.

## Main Risks

- Overconfident auto-answering can push the key traversal down the wrong branch.
- YOLO part masks may be missing, mislabeled, or low quality.
- `Underside` is known to be the weakest YOLO class, so underside-derived traits need conservative confidence thresholds.
- Some `key.xml` questions describe non-visual traits and must remain user questions.
- More trait fields can make downstream prompts and database comparison noisier if confidence and provenance are not preserved.

## Expected Result

After the refactor, the extractor should no longer treat YOLO as just a foreground mask provider. It should use YOLO class labels as part-level evidence, extract traits from the correct mushroom region, report confidence and source for each important trait, and answer more of `key.xml` only when the image evidence is strong enough.
