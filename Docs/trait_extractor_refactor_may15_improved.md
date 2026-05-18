# Trait Extractor Refactor Plan (Improved) - May 15, 2026

## Purpose

The trait extractor originally analysed the whole image and relied on classical image processing, including Otsu thresholding, to infer mushroom shape and appearance. After introducing YOLOv8 segmentation, mushroom isolation is no longer the extractor's main responsibility. YOLO now supplies masks for mushroom parts, so the extractor should analyze the segmented mushroom regions instead of the full photo.

This refactor is needed because the current implementation is only partially mask-aware. It can use a segmentation mask, but in the unified pipeline it selects one "best" YOLO instance and extracts all traits from that single mask. If the highest-confidence instance is a stem or underside, cap traits may be computed from the wrong part. The extractor should be class-aware: cap traits should come from the cap mask, stem traits from the stem mask, underside traits from the underside mask, and coral traits from the coral mask.

> **⚠️ Issue:** The original plan targeted `_extract_traits_masked()` in `unified_pipeline.py` while ignoring the standalone `extract()` entry point in `visual_trait_extractor.py`. `extract()` still runs its own segmentation, picks a single best instance (`sel_idx`), and applies one mask globally. Any direct caller of `extract()` — including the proposed benchmark tests and baseline snapshot script — will bypass the new part-aware logic entirely. Additionally, the unified pipeline runs YOLO once in `_segment()` and then `extract()` runs it **again** internally, doubling inference cost.
>
> **✅ Fix:** Standardize `extract()` to accept optional pre-computed part masks and perform its own class-aware routing when masks are provided. This eliminates double-segmentation and guarantees one consistent behaviour path for all callers.

---

## Pre-Phase 0 — Baseline Capture (MANDATORY)

> **⚠️ Issue:** The original plan described a baseline snapshot script but did not explicitly mandate running it **before** any code changes. Once Phase 1 modifies `visual_trait_extractor.py`, the old behaviour is overwritten and a clean before/after comparison becomes impossible.
>
> **✅ Fix:** Add an explicit Phase 0 that must be completed, the JSON artifact committed, and tagged before any refactoring code is written.

**Tasks:**
1. Run the baseline snapshot script (see Baseline Snapshot Script section below).
2. Save the output to `artifacts/trait_extractor_baseline_may15.json`.
3. Commit the artifact with a clear message: `git add artifacts/ && git commit -m "baseline: trait extractor pre-refactor snapshot"`.
4. Do not begin Phase 1 until the baseline is preserved.

---

## Current State

- `models/visual_trait_extractor.py` already contains masked variants for colour, shape, texture, and brightness analysis.
- `models/mushroom_segmenter.py` parses YOLOv8 segmentation results and exposes class names: `Cap`, `Coral`, `Stem`, and `Underside`.
- `models/unified_pipeline.py` calls masked extraction, but `_extract_traits_masked()` currently chooses one best instance and uses that mask for all trait extraction.
- `data/raw/key.xml` contains many traits that are visually observable, but the current auto-answering in `models/key_tree_traversal.py` uses only a small subset of colour, shape, and ridge signals.
- `models/trait_database_comparator.py` compares visible traits against `species_traits.xml`. The XML does not yet contain fields for the new part-aware traits (`hymenophore_type`, `stem_ring`, `coral_branching`, etc.).

---

## Target Design

The improved extractor should introduce a YOLO part aggregation layer that converts raw YOLO instances into stable, named part masks:

```python
{
    "cap": {"mask": ..., "confidence": ..., "bbox": ...},
    "stem": {"mask": ..., "confidence": ..., "bbox": ...},
    "underside": {"mask": ..., "confidence": ..., "bbox": ...},
    "coral": {"mask": ..., "confidence": ..., "bbox": ...},
    "whole": {"mask": union_of_all_parts, "confidence": ..., "bbox": ...},
}
```

> **⚠️ Issue:** The original plan used lowercase keys (`cap`, `stem`, ...) while `CLASS_NAMES` in `mushroom_segmenter.py` returns title-case names (`Cap`, `Stem`, ...). A silent mapping bug is likely if this is not made explicit.
>
> **✅ Fix:** Define a normalisation map in the `YoloPartMasks` helper:
> ```python
> _NAME_NORMALIZE = {"cap": "Cap", "stem": "Stem", "underside": "Underside", "coral": "Coral"}
> ```
> The helper should group by the raw `class_name`, then expose lowercase keys for downstream consumers.

Extraction should then route by part:

- Use `Cap` mask for cap colour, cap shape, cap texture, cap margin, and cap surface markings.
- Use `Stem` mask for stem colour, stem proportions, ring detection, bulbous base detection, stem surface texture, scabers/tofsar, and network pattern.
- Use `Underside` mask for gills, pores, ridges, teeth, underside colour, and decurrent/attachment cues.
- Use `Coral` mask for coral-like branching, branch tips, branch colour, and cauliflower-like flattened folds.
- Use the union mask only for whole-fruiting-body fallback traits and quality checks.

### Standardized API

> **✅ Fix:** Change the public entry point so it can receive pre-computed masks from the unified pipeline:

```python
def extract(
    image_bytes: bytes,
    part_masks: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Full Step-1 analysis.

    Args:
        image_bytes: Raw image bytes.
        part_masks: Optional dict from YoloPartMasks. If provided, segmentation
            is skipped and these masks are used directly. If None, the extractor
            runs its own YOLO segmentation internally (legacy standalone mode).
    """
```

This removes the double-YOLO cost in the unified pipeline and guarantees that benchmark tests exercise the same code path as production.

---

## Morphology Cases

The extractor should explicitly handle four routed morphology cases:

1. `puffball`
   - Round or pear-shaped body.
   - Usually no visible stem, cap, or underside separation.
   - Extract whole-body colour, roundness, pear-shape score, surface warts/spines/smoothness, and clustered-growth cue.

2. `coral`
   - YOLO detects `Coral`, or the mask has a strong branching skeleton.
   - Extract branching count, branch thickness, branch tip colour, finger-like vs cauliflower-like shape, and overall colour.

3. `classical_convex`
   - Cap plus stem and/or underside are detected.
   - Cap contour is dome-like, rounded, bell-shaped, or flat-convex.
   - Extract cap colour/shape/texture, stem colour/ring, underside type, and relative geometry.

4. `classical_concave`
   - Cap plus stem and/or underside are detected.
   - Cap or whole silhouette suggests funnel-shaped, depressed, trumpet-like, or concave form.
   - Extract cap depression/funnel score, ridges/folds, decurrent structures, stem colour, and dark/trumpet-like whole-body traits.

> **⚠️ Issue:** The original plan listed `classical_convex` and `classical_concave` but `detect_case()` in `mushroom_segmenter.py` only returns `"classical"`. There was no rule for splitting the two cases.
>
> **✅ Fix:** Add an explicit derivation strategy. After the cap shape is extracted, map it to the morphology case:

```python
def derive_morphology_case(detected_parts: Set[str], cap_shape: str) -> str:
    if "Coral" in detected_parts:
        return "coral"
    # Puffball: only Cap visible, no Stem/Underside, and high circularity
    if detected_parts == {"Cap"}:
        return "puffball"
    # Classical split based on contour shape
    if cap_shape in {"funnel-shaped", "depressed"}:
        return "classical_concave"
    if cap_shape in {"convex", "flat", "bell-shaped"}:
        return "classical_convex"
    return "uncertain"
```

> **⚠️ Issue:** `detect_case()` required Cap-only in **both** above and below photos to trigger `puffball`. If the below photo is missing or blurry, the case falls through to `uncertain` even when the above photo is clearly a round puffball.
>
> **✅ Fix:** Relax the rule. Allow single-photo puffball detection when the visible photo contains only `Cap` with circularity > 0.75 and area ratio within puffball range. The unified pipeline can merge the case from whichever photo is more confident.

---

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

    "morphology_case": "puffball|coral|classical_convex|classical_concave|uncertain",
    "detected_parts": ["Cap", "Stem", "Underside"],
    "hymenophore_type": "gills|pores|ridges|teeth|unknown",
    "cap_color": "...",
    "stem_color": "...",
    "underside_color": "...",
    "cap_surface": "smooth|scaly|warty|spiny|hairy|viscid_unknown",
    "stem_ring": "present|absent|unknown",
    "stem_surface": "smooth|fibrous|scabers|network|unknown",
    "coral_branching": "finger_like|cauliflower_like|unknown",
    "puffball_surface": "smooth|warty|spiny|unknown",
    "trait_confidence": {
        "cap_color": 0.0,
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

> **⚠️ Issue:** The original plan did not address `_merge_traits()` in `unified_pipeline.py`. When Phase 2 adds `cap_color`, `stem_color`, `cap_surface`, etc., the merge logic must know which photo to prefer for which part (above photo for cap traits, below photo for underside traits).
>
> **✅ Fix:** Extend `_merge_traits()` with a per-part photo preference table:

```python
_PHOTO_PREFERENCE = {
    "cap_color": "above",
    "cap_shape": "above",
    "cap_surface": "above",
    "underside_color": "below",
    "hymenophore_type": "below",
    "stem_color": "below",   # stem is often clearer in below photo
    "stem_ring": "below",
    "coral_branching": "above",
}
```

For traits with a preference, take the preferred photo's value if it is not `unknown`/`None`. For other traits, keep the existing "first non-empty wins" logic.

---

## Algorithm Improvements

### Shared Mask Handling

- Normalize all YOLO masks to binary `uint8`.
- Remove small components per class.
- Keep the highest-confidence instance per class, but allow multiple instances when detecting clustered growth.
- Compute mask quality per part: area ratio, fragmentation, hole ratio, boundary irregularity, confidence, and whether the mask touches image borders.
- Fall back to the whole union mask only when the intended class mask is missing or fails quality gates.

> **⚠️ Issue:** "Allow multiple instances when detecting clustered growth" was underspecified. There was no algorithm for merging masks or computing combined confidence.
>
> **✅ Fix:** Define the clustered-growth rule explicitly:
> - If two or more instances of the same class are detected and their bboxes overlap (IoU > 0.05) or are within 50 px, merge their masks with `np.logical_or`.
> - Combined confidence = weighted average by mask area.
> - Combined bbox = enclosing rectangle of all merged masks.
> - Flag `clustered_growth = True` in the output when more than one non-overlapping fruit body is detected.

### Colour Extraction

- Compute cap colour from `Cap`, stem colour from `Stem`, underside colour from `Underside`, and whole colour from the union mask.
- Use masked pixel sampling instead of full-image resizing.
- Preserve existing colour ratios but compute them over the relevant mask area.
- Add support for colour groups needed by `key.xml`: yellow, orange, red, brown, grey-brown, black/dark-grey, white/grey-white/yellow-white, violet/lilac tips.

> **⚠️ Issue:** `analyse_colours_masked` uses `np.random.choice(..., replace=False)` without a fixed seed. Benchmark baselines will not be reproducible across runs.
>
> **✅ Fix:** Replace random sampling with deterministic strided sampling or pass `random_state=42`:
> ```python
> if len(pixels) > 4096:
>     step = len(pixels) // 4096
>     idx = np.arange(0, len(pixels), step)[:4096]
>     pixels = pixels[idx]
> ```

### Shape Extraction

- Replace Otsu-based cap shape in the YOLO path with contour analysis on the cap mask.
- Use aspect ratio, circularity, convexity, contour solidity, upper-outline curvature, and central-depression score.
- Classify cap shape as `convex`, `flat`, `bell-shaped`, `funnel-shaped`, `depressed`, `wavy`, `irregular`, or `unknown`.
- For puffballs, compute roundness and pear-shape score from the whole mask.
- For morel/false-morel-like cases, detect honeycomb/network vs brain-like folds using high-frequency texture on the cap/whole mask.

> **⚠️ Issue:** The existing `analyse_shape` / `analyse_shape_masked` heuristics have overlapping branches. For example, `aspect_ratio < 0.7` (bell-shaped) is checked before `0.5 < aspect_ratio < 1.0 and circularity < 0.65` (funnel-shaped). A funnel-shaped mushroom with aspect ratio 0.6 will be misclassified as `bell-shaped` because the first branch catches it.
>
> **✅ Fix:** Redefine the shape decision tree with mutually exclusive conditions and add synthetic contour unit tests:

```python
if circularity > 0.80 and 0.8 <= aspect_ratio <= 1.3:
    cap_shape = "convex"
elif aspect_ratio >= 1.6 and circularity < 0.6:
    cap_shape = "flat"
elif aspect_ratio <= 0.5 and circularity >= 0.70:
    cap_shape = "bell-shaped"
elif 0.5 < aspect_ratio < 0.9 and circularity < 0.60:
    cap_shape = "funnel-shaped"
elif aspect_ratio >= 0.9 and circularity < 0.45:
    cap_shape = "wavy"
elif aspect_ratio < 0.9 and circularity >= 0.60:
    cap_shape = "bell-shaped"   # narrow but smooth cone
else:
    cap_shape = "unknown"
```

### Texture and Surface Extraction

> **⚠️ Issue:** The original plan identified that `analyse_texture_masked` divides edge density by the full image area instead of the mask area, but it buried this fix in narrative text without assigning it to a phase.
>
> **✅ Fix:** Schedule this explicitly in Phase 1. The corrected implementation:

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

### Stem Extraction

Use the `Stem` mask where available.

- Stem colour and brightness.
- Stem aspect ratio and thickness.
- Ring detection by looking for a horizontal band or high-contrast annulus around the upper stem.
- Bulbous base from lower-stem width compared with mid-stem width.
- Scabers/tofsar from dark small blob density on the stem.
- Network pattern from crossing line structures on the stem surface.

### Coral Extraction

Use the `Coral` mask where available.

- Skeletonize the coral mask.
- Count branch endpoints and junctions.
- Estimate branch thickness and branch density.
- Classify `finger_like` vs `cauliflower_like`:
  - finger-like: separated upright branches, many endpoints, narrow branches.
  - cauliflower-like: dense flattened lobes, high boundary complexity, broad folded sheets.
- Compare tip colour against base colour for violet-tipped or yellow-branched cases.

---

## Using YOLO Classes to Enhance Key Traversal

YOLO classes should directly improve the root `key.xml` question: `Hur ser svampen ut?`

Suggested mapping:

- `Coral` detected with sufficient confidence -> `Den är busklik med många grenar`
- Puffball case -> `Päronformad eller rund`
- `Underside` plus ridge/fold detector -> `Undersidan har åsar eller ådror`
- `Underside` plus gill detector -> `Undersidan har skivor`
- `Underside` plus pore detector -> `Undersidan har rör`
- `Underside` plus tooth detector -> `Undersidan har taggar`
- Morel/false-morel texture -> `Vindlingar eller nätmonster`

> **⚠️ Issue:** The original plan contained Swedish spelling errors that would cause exact-match failures against `key.xml` answers.
>
> **✅ Fix:** The strings above have been corrected to match `key.xml` exactly.

The implementation should avoid blindly mapping `Underside` to one option. YOLO says that an underside exists; the extractor still needs texture analysis to choose gills, pores, ridges, or teeth.

---

## Traits From key.xml That Image Analysis Can Fill

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

---

## Implementation Phases

### Phase 0 — Baseline (Pre-Requisite)

1. Run the baseline snapshot script (see Baseline Snapshot Script section).
2. Commit `artifacts/trait_extractor_baseline_may15.json`.
3. Verify the manifest contains 57 specimens and all image paths resolve.
4. **Do not proceed until the baseline artifact is preserved.**

### Phase 1 — Part-Aware Mask Routing

- Add a `YoloPartMasks` helper or equivalent function.
- Group instances by class name; normalize title-case (`Cap`, `Stem`, ...) to lowercase keys (`cap`, `stem`, ...).
- Merge masks for clustered growth using `np.logical_or` when bboxes overlap or are within 50 px.
- Produce per-part masks plus a whole union mask.
- **Fix masked edge-density denominator** so it divides by mask area, not full image area.
- **Fix colour sampling** to use deterministic strided sampling instead of `np.random.choice`.
- Replace single-best-instance logic in `extract()` and `_extract_traits_masked()`.
- Preserve existing output keys to avoid breaking the API and tests.
- Update `detect_case()` to use the new `derive_morphology_case()` strategy (convex/concave split + relaxed puffball rule).

### Phase 2 — Part-Specific Trait Functions

- Add `analyse_cap_traits(bgr, cap_mask)`.
- Add `analyse_stem_traits(bgr, stem_mask)`.
- Add `analyse_underside_traits(bgr, underside_mask, stem_mask=None)`.
- Add `analyse_puffball_traits(bgr, whole_mask)`.
- Add `analyse_coral_traits(bgr, coral_mask)`.
- **Fix shape heuristic overlaps** (bell-shaped vs funnel-shaped) with the mutually exclusive decision tree above.
- Add confidence scores and source metadata for each trait.
- Provide a straw-man confidence formula so thresholding in Phase 3 is concrete:

```python
def trait_confidence(mask_quality: Dict[str, float], detector_score: float) -> float:
    """
    mask_quality keys: area_ratio, fragmentation, hole_ratio,
                       boundary_irregularity, yolo_confidence
    """
    area_q = min(mask_quality["area_ratio"] / 0.05, 1.0)
    frag_q = max(0.0, 1.0 - (mask_quality["fragmentation"] - 1) * 0.25)
    hole_q = max(0.0, 1.0 - mask_quality["hole_ratio"] / 0.10)
    border_penalty = 0.9 if mask_quality.get("touches_border", False) else 1.0
    return float(area_q * frag_q * hole_q * border_penalty * mask_quality["yolo_confidence"] * detector_score)
```

### Phase 3 — Key.xml Mapping

- Add a `derive_key_answers(visible_traits)` function.
- Extend `_try_auto_answer()` to use explicit structured traits such as `hymenophore_type`, `morphology_case`, `stem_ring`, `cap_surface`, and `coral_branching`.
- Use the corrected Swedish answer strings listed above.
- Keep conservative thresholds. If confidence is low, return `None` and let the UI/user answer.

### Phase 4 — Database Comparator Improvements

> **⚠️ Issue:** The original plan assumed `TraitDatabaseComparator` could be extended without touching the underlying data source. `species_traits.xml` does not yet contain fields for `hymenophore_type`, `stem_ring`, `coral_branching`, etc.
>
> **✅ Fix:** Before modifying the comparator, update `data/raw/species_traits.xml` (and the loader in `data/dataset_utils.py` if necessary) to include the new trait categories. Only then extend `_compare_visible_to_db` and `_TRAIT_WEIGHTS`.

- Extend `TraitDatabaseComparator` beyond the current comparable traits.
- Add support for:
  - `hymenophore_type`
  - `stem_ring`
  - `stem_surface`
  - `cap_surface`
  - `morphology_case`
  - `underside_color`
  - `puffball_surface`
  - `coral_branching`
- Weight part-aware traits higher when their YOLO class confidence is high.

### Phase 5 — Tests and Benchmarks

- Add synthetic tests for part routing:
  - cap colour is computed from cap mask, not stem/underside.
  - underside texture determines gills/pores/ridges/teeth.
  - coral class routes to coral extraction.
  - puffball cap-only round mask routes to puffball extraction.
- Add regression tests for current benchmark scenarios: `puffball`, `coral`, `confusing`, and `easy`.
- Re-run comparative benchmark and compare tree coverage and database score before/after.

---

## Verification Without the Benchmark Runner

The benchmark runner is not required to verify the trait extractor improvement. The repository already contains the benchmark image manifest at `benchmarks/evaluation_manifest.csv`, and the referenced images are stored under `data/raw/Benchmark/`. Verification should use those same images directly through pytest or a small local diagnostic script.

At the time this plan was written, the manifest contains 57 specimens and all referenced image paths exist locally:

- `confusing`: 22 specimens
- `ood`: 17 specimens
- `coral`: 6 specimens
- `puffball`: 5 specimens
- `easy`: 5 specimens
- `edge_case`: 2 specimens

The verification method is a before/after comparison:

1. Run the current trait extractor on every `above` and `below` image listed in `benchmarks/evaluation_manifest.csv`.
2. Save the current outputs as a baseline JSON artifact, for example `artifacts/trait_extractor_baseline_may15.json`.
3. Implement the part-aware extractor.
4. Run the new extractor on the same manifest images.
5. Compare old vs new outputs using explicit trait-level metrics rather than final species accuracy.

This is important because final identification accuracy depends on several downstream systems: YOLO segmentation, tree traversal, CNN, database comparison, and LLM synthesis. The trait extractor should be judged first on whether it produces better visual traits from the same images.

---

## Trait-Level Verification Metrics

The following checks should be computed from the manifest notes, scenario labels, species IDs, and expected visual categories:

- `mask_source_correctness`: cap traits came from cap masks, stem traits from stem masks, underside traits from underside masks, and coral traits from coral masks.
- `morphology_case_accuracy`: expected scenario or notes agree with extracted `morphology_case`.
- `hymenophore_accuracy`: images noted as `gills`, `pores`, `folds`, `teeth`, or `ridges` produce the corresponding `hymenophore_type`.
- `puffball_accuracy`: `puffball` scenario images produce `morphology_case = puffball` and do not invent gills, pores, or stem traits.
- `coral_accuracy`: `coral` scenario images produce `morphology_case = coral` and expose `coral_branching`.
- `cap_colour_stability`: known colour groups in notes/species are retained, such as yellow chanterelles, dark trumpets, brown boletes, red fly agaric, and white amanita.
- `tree_auto_answer_coverage`: the number of `key.xml` questions answered from image traits increases, but only when confidence passes a conservative threshold.
- `false_auto_answer_rate`: visually unsupported questions remain unanswered; this should not increase.
- `db_comparable_trait_count`: the database comparator receives more comparable traits per specimen after the refactor.

The refactor should be considered improved only if it increases useful trait coverage without increasing confident wrong auto-answers.

---

## Pytest Fallback Test Harness

Add a pytest file that loads the manifest directly instead of running the full benchmark:

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

These tests should initially be added as expected-failing or skipped until the new output schema is implemented. Once the refactor is complete, they become regression tests.

---

## Baseline Snapshot Script

Add a small script or pytest helper that stores extractor output for the current implementation:

```python
import json
from pathlib import Path

from benchmarks.manifest import ManifestDataset
from models.visual_trait_extractor import extract

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "benchmarks" / "evaluation_manifest.csv"
OUT = PROJECT_ROOT / "artifacts" / "trait_extractor_baseline_may15.json"

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

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
```

The baseline artifact makes verification concrete. The new extractor can be compared against this file with simple counters:

- how often morphology case changed from `uncertain` to the expected case;
- how often `hymenophore_type` became available;
- how often cap/stem/underside colours changed due to correct part masks;
- how many extra `key.xml` questions can be answered;
- how many new answers conflict with manifest notes.

---

## Main Risk

The main risk is overconfident auto-answering. Some `key.xml` questions describe physical or ecological traits that cannot be safely inferred from an image. The extractor should report only observable visual evidence, attach confidence values, and leave uncertain or non-visual questions unanswered.

> **⚠️ Additional risk:** Because the original `extract()` ran YOLO internally, the unified pipeline performed two full YOLO inferences per photo. The refactor must not accidentally triple that cost by adding extra per-part model calls. All new analysis should be classical-CP operations (contours, KMeans, Hough, morphology) on the already-generated masks.
>
> **✅ Mitigation:** The standardized `extract(image_bytes, part_masks=None)` API lets the unified pipeline pass masks directly, eliminating redundant YOLO calls. Document that no new neural-network models are introduced in this refactor.

---

## Expected Result

After the refactor, the trait extractor should no longer treat YOLO as just a foreground mask provider. It should use YOLO's class labels as part-level evidence, extract traits from the correct mushroom region, and produce enough structured visual traits to answer more of `key.xml` without forcing unreliable guesses.
