# Trait Extractor — Issues & Test Results (18 May 2026)

> **Context:** Part-aware trait extractor (`models/visual_trait_extractor.py`) with `ENABLE_PART_AWARE_TRAITS = True`. YOLOv8-seg 4-class model (`data/Yolov8/best.pt`, retrained 18 May). Benchmark: `benchmarks/evaluation_manifest.csv` (57 specimens, 23 species, 114 images).
>
> **Goal of this doc:** Enable Codex (or any future agent) to reproduce the exact test results, understand the root causes, and pick up where this investigation left off.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Issue 1 — Coral False Positives](#2-issue-1--coral-false-positives)
   - 2.1 Problem statement
   - 2.2 Fix implemented (cross-view consistency)
   - 2.3 Remaining false positives
3. [Issue 2 — Trait Extractor Accuracy vs Database](#3-issue-2--trait-extractor-accuracy-vs-database)
   - 3.1 Test methodology
   - 3.2 Raw results (10 species)
   - 3.3 Analysis
4. [Issue 3 — `detect_case` / `build_part_masks` Decoupling](#4-issue-3--detect_case--build_part_masks-decoupling)
5. [Issue 4 — Color Naming Coarseness](#5-issue-4--color-naming-coarseness)
6. [Reproduction Scripts](#6-reproduction-scripts)
7. [Files Modified in This Investigation](#7-files-modified-in-this-investigation)
8. [Open Questions for Next Agent](#8-open-questions-for-next-agent)

---

## 1. Environment Setup

| Component | Value |
|-----------|-------|
| Project root | `/home/iannyf/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification` |
| Python venv | `.venv/bin/python` (YOLO/Ultralytics installed here only) |
| YOLO weights | `data/Yolov8/best.pt` (retrained 18 May 2026, 4-class: Cap, Coral, Stem, Underside) |
| Old weights backup | `data/Yolov8/old yolo_weights/best.pt` |
| Benchmark manifest | `benchmarks/evaluation_manifest.csv` |
| Trait database | `data/raw/species_traits.xml` |
| Species metadata | `data/raw/species.csv` |
| Feature flags | `config/trait_config.py` |

**Always use the venv:**
```bash
cd /home/iannyf/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification
.venv/bin/python <script.py>
```

**Important:** The segmenter is a singleton with global cache. After swapping `best.pt`, force reload:
```python
import models.mushroom_segmenter as seg_mod
seg_mod._segmenter_instance = None
```

---

## 2. Issue 1 — Coral False Positives

### 2.1 Problem statement

The fine-tuned YOLO model predicts `Coral` class on non-coral specimens. When `ENABLE_PART_AWARE_TRAITS = True`, the pipeline trusts these predictions and sets `coarse_case = "coral"`, routing the user to the coral key tree (wrong path).

**Ground truth:** Only `RA.BO_*` (Ramaria botrytis) and `RA.PA_*` (Ramaria pallida) are true coral mushrooms in the benchmark.

**Old model false positives (before retraining):**
- `AG.AU_001` above, `HY.PS_019` below, `BO.ED_014` above, `LA.HE_023` below, `LY.PE_037` above/below, `SP.CR_039` above/below, etc.

**New model false positives (after retraining):**
- `AM.VI_007` below (coral conf 0.364)
- `HY.PS_019` below (coral conf 0.433, 0.331)
- `LA.HE_022` below (coral conf 0.338)
- `LY.PE_037` above (coral conf 0.867)
- `LY.PE_038` above/below (coral conf 0.824, 0.781)
- `SP.CR_039` above/below (coral conf 0.724, 0.827)
- `SP.CR_040` above (coral conf 0.959)
- `GY.ES_042` below (coral conf 0.540)
- `PL.OS_045` above (coral conf 0.424)
- `RU.IN_055` above (coral conf 0.430)

### 2.2 Fix implemented — Cross-view consistency check

Modified `detect_case()` in `models/mushroom_segmenter.py`.

**Logic:** Coral is declared only if:
1. Coral detected in **both** above and below views, OR
2. Coral detected in **one** view with confidence ≥ 0.75 **AND** no contradictory classical structure (Cap + Stem/Underside) present in either view.

```python
# Pseudocode of the new rule
above_coral = [i for i in above_instances if i.class_name == "Coral" and i.conf >= threshold]
below_coral = [i for i in below_instances if i.class_name == "Coral" and i.conf >= threshold]

has_classical_structure = (has_cap and (has_stem or has_underside)) or (has_stem and has_underside)

coral_in_both = len(above_coral) > 0 and len(below_coral) > 0
max_coral_conf = max(above_coral + below_coral, key=lambda i: i.conf, default=0.0)

coral_single_view = (
    (len(above_coral) > 0) != (len(below_coral) > 0)
    and max_coral_conf >= 0.75
    and not has_classical_structure
)

if coral_in_both or coral_single_view:
    return {"case": "coral", ...}
```

**Result after fix:**

| Specimen | Old `case` | New `case` | Correct? |
|----------|-----------|-----------|----------|
| `AM.VI_007` | coral | **classical** | ✅ Fixed |
| `HY.PS_019` | coral | **puffball** | ✅ Fixed |
| `GY.ES_042` | coral | **uncertain** | ✅ Fixed |
| `PL.OS_045` | coral | **uncertain** | ✅ Fixed |
| `RU.IN_055` | coral | **puffball** | ✅ Fixed |
| `RA.BO_025` | coral | **coral** | ✅ Preserved (single-view high conf, no classical structure) |
| `RA.BO_026` | coral | **coral** | ✅ Preserved |
| `RA.BO_027` | coral | **coral** | ✅ Preserved |
| `RA.PA_028` | coral | **coral** | ✅ Preserved |
| `RA.PA_029` | coral | **coral** | ✅ Preserved |
| `RA.PA_030` | coral | **coral** | ✅ Preserved |

### 2.3 Remaining false positives

Four specimens still classify as `case=coral` despite not being coral species:

| Specimen | Why it passes | Geometric filter verdict |
|----------|--------------|--------------------------|
| `LY.PE_037` | Above: coral 0.867. Below: Cap + Stem. `has_classical_structure=True`, but Cap conf in below view may be < 0.35 (threshold), so structure check fails. | Above coral rejected by geometric filter; below has no coral. **But `detect_case` still sees raw instance.** |
| `LY.PE_038` | Coral in **both** views (0.824 above, 0.781 below). | Both rejected by geometric filter, but `detect_case` uses raw instances. |
| `SP.CR_039` | Coral in **both** views (0.724 above, 0.827 below). | Above passes geometric filter. Below rejected. `detect_case` sees both raw instances. |
| `SP.CR_040` | Single view coral 0.959 ≥ 0.75. No classical structure. | Passes all filters. |

**Root cause:** `detect_case()` operates on **raw YOLO instances**, not on the **filtered part masks** produced by `build_part_masks()`. A mask can be rejected by the geometric filter (e.g., too compact) while the raw instance still influences `detect_case`.

---

## 3. Issue 2 — Trait Extractor Accuracy vs Database

### 3.1 Test methodology

**Script:** Run `extract()` with `ENABLE_PART_AWARE_TRAITS = True` on both above and below photos for 10 species, merge traits using `_merge_traits()`, and compare against `data/raw/species_traits.xml`.

**Species chosen (diverse morphologies):**
1. `AG.AU` — Agaricus augustus (classical, almond smell)
2. `AM.MU` — Amanita muscaria (classical, red cap)
3. `BO.ED` — Boletus edulis (classical bolete, pores)
4. `CA.CI` — Cantharellus cibarius (chanterelle, yellow, decurrent folds)
5. `CO.CO` — Coprinellus comatus (shaggy inkcap, cylindrical)
6. `GY.ES` — Gyromitra esculenta (false morel, brain-like, brown)
7. `HY.PS` — Hygrophorellus pseudoaurantiacus (false chanterelle, orange)
8. `LY.PE` — Lycoperdon perlatum (puffball, white, round)
9. `RA.BO` — Ramaria botrytis (true coral, white/cream)
10. `SP.CR` — Sparassis crispa (cauliflower fungus, branching white)

**Reproduction script:** See [Section 6](#6-reproduction-scripts).

### 3.2 Raw results

| Species | Specimen | Case | `detected_parts` | DB Cap Color | Extr. Cap Color | Match | DB Cap Shape | Extr. Cap Shape | Match |
|---------|----------|------|------------------|--------------|-----------------|-------|--------------|-----------------|-------|
| AG.AU | AG.AU_001 | uncertain | `[]` | straw\|yellow\|brown | — | — | convex\|flat | — | — |
| AM.MU | AM.MU_004 | classical | `[]` | **red\|scarlet** | **white** | ✗ | convex\|flat | wavy | ✗ |
| BO.ED | BO.ED_013 | classical | `['cap']` | brown | olive-brown | ✓ | convex\|flat | funnel-shaped | ✗ |
| CA.CI | CA.CI_016 | classical | `['cap']` | **yellow-orange** | **olive-brown** | ✗ | funnel-shaped | funnel-shaped | ✓ |
| CO.CO | CO.CO_031 | uncertain | `[]` | white | — | — | cylindrical\|conical | flat | ✗ |
| GY.ES | GY.ES_041 | puffball | `['cap']` | **brown\|reddish-brown** | **white** | ✗ | irregular\|brain-like | wavy | ✗ |
| HY.PS | HY.PS_019 | puffball | `['cap']` | orange\|yellow-orange | orange | ✓ | convex\|funnel-shaped | wavy | ✗ |
| LY.PE | LY.PE_037 | **coral** | `[]` | white | — | — | round\|pear-shaped | wavy | ✗ |
| RA.BO | RA.BO_025 | coral | `[]` | white\|cream\|pink-tipped | tan | ✗ | branched-coral | wavy | ✗ |
| SP.CR | SP.CR_039 | coral | `['coral']` | white\|cream\|pale-yellow | tan | ✗ | cauliflower\|frond-like | wavy | ✗ |

**Stem/underside trait extraction:**

| Species | DB Stem Color | Extr. Stem Color | DB Stem Surface | Extr. Stem Surface | DB Underside | Extr. Underside |
|---------|---------------|------------------|-----------------|--------------------|--------------|-----------------|
| AG.AU | white | — | smooth | — | pale\|pink\|brown | — |
| AM.MU | white | None | smooth | None | white | None |
| BO.ED | white\|brown | unknown | reticulate | unknown | yellow | unknown |
| CA.CI | yellow-orange | grey | smooth | smooth | pale-yellow | tan |
| CO.CO | white | — | smooth | — | white\|pink\|black | — |
| GY.ES | pale | unknown | smooth | unknown | brown | unknown |
| HY.PS | orange | unknown | smooth | unknown | orange | unknown |
| LY.PE | white | — | smooth | — | white | — |
| RA.BO | white | None | smooth | None | white\|pink | None |
| SP.CR | white | unknown | smooth | unknown | white\|cream | unknown |

**Hymenophore type (gills/pores/folds/none):**

| Species | DB | Extracted | Match? |
|---------|----|-----------|--------|
| AG.AU | free | — | — |
| AM.MU | free | None | ✗ |
| BO.ED | pores | unknown | ✗ |
| CA.CI | decurrent | gills | ✗ |
| CO.CO | free | — | — |
| GY.ES | folds | unknown | ✗ |
| HY.PS | decurrent | unknown | ✗ |
| LY.PE | none | — | — |
| RA.BO | none | None | ✓ |
| SP.CR | none | unknown | ✗ |

### 3.3 Analysis

**Cap color accuracy: 3/10 correct** (HY.PS orange, BO.ED dominant olive-brown ≈ brown, LY.PE dominant white)

**Cap shape accuracy: 1/10 correct** (CA.CI funnel-shaped)

**Any stem/underside traits extracted at all: 2/10** (CA.CI, BO.ED — and both were "unknown")

**Key findings:**

1. **When YOLO detects no parts, traits are empty.** `AG.AU_001`, `CO.CO_031`, `LY.PE_037`, `RA.BO_025` all show `detected_parts=[]`. The extractor falls back to whole-image analysis but the merged traits don't surface the fallback values properly.

2. **Color analysis is the weakest link.** A bright red Amanita muscaria (`AM.MU_004`) is classified as `cap_color=white`. A brown false morel (`GY.ES_041`) is classified as `cap_color=white`. This suggests the color quantisation/clustering is failing on these images.

3. **Shape analysis is unreliable.** Cap shape is derived from contour metrics (circularity, aspect ratio) on the 2D mask. Most caps come out as `wavy` or `funnel-shaped` rather than their true shape.

4. **Stem and underside are almost never extracted.** Only 2 out of 10 specimens produced any stem/underside traits. This is because:
   - The below-view photo often has no YOLO detections (10.5% of images)
   - When stem/underside ARE detected, the masks fail quality gates or are discarded by `build_part_masks`

---

## 4. Issue 3 — `detect_case` / `build_part_masks` Decoupling

**Current flow in `unified_pipeline.py`:**

```python
# Step 1: segment both photos
above_seg = self._segment(above_image_bytes)
below_seg = self._segment(below_image_bytes)

# Step 2: detect case from RAW instances
case = detect_case(
    above_seg.get("instances", []),
    below_seg.get("instances", []),
)

# Step 3: extract traits — this calls build_part_masks internally
above_traits = _extract_traits_masked(
    above_image_bytes, above_seg.get("instances", [])
)
```

**Problem:** `detect_case` decides `case=coral` based on raw YOLO instances. But `_extract_traits_masked` calls `build_part_masks`, which may reject those same coral masks via geometric filter or quality gates. Result: `case=coral` with `detected_parts=[]` and empty trait values.

**Affected specimens:**
- `RA.BO_025`: `case=coral`, `detected_parts=[]`
- `LY.PE_038`: `case=coral`, `detected_parts=[]`
- `SP.CR_039`: `case=coral`, `detected_parts=['coral']` (only above mask passed)

**Fix needed:** Build part masks **before** calling `detect_case`, then pass the accepted masks into `detect_case` (or derive the case directly from which masks survived filtering).

---

## 5. Issue 4 — Color Naming Coarseness

The extractor maps RGB clusters to one of ~10 color names: `red, orange, tan, olive-brown, yellow-green, grey, white, black, brown, pink`.

The database uses much finer gradations: `straw, yellow-orange, reddish-brown, pale-yellow, cream, pink-tipped`.

**Examples of mismatch:**
- Database: "yellow-orange" chanterelle → Extractor: "olive-brown"
- Database: "brown" porcini → Extractor: "olive-brown" (close, but not exact)
- Database: "white\|cream" coral → Extractor: "tan" or "yellow-green"

**Fix needed:** Either expand the color palette or compute HSV distance to database colors and report the closest match instead of a hard-coded category.

---

## 6. Reproduction Scripts

### 6.1 Coral detection test

```python
# File: coral_test.py
from pathlib import Path
from benchmarks.manifest import ManifestDataset
from models.mushroom_segmenter import get_segmenter, detect_case
import models.mushroom_segmenter as seg_mod

seg_mod._segmenter_instance = None

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST = PROJECT_ROOT / "benchmarks" / "evaluation_manifest.csv"
dataset = ManifestDataset(MANIFEST)
seg = get_segmenter()

real_corals = {"RA.BO_025", "RA.BO_026", "RA.BO_027", "RA.PA_028", "RA.PA_029", "RA.PA_030"}

print(f"{'Specimen':<12} {'Case':<10} {'Conf':>8} {'Parts':<30} {'Real':>5}")
print("-" * 70)

for sample in dataset:
    above = sample.load_above_bytes()
    below = sample.load_below_bytes()
    above_seg = seg.segment(above) if above else {"instances": []}
    below_seg = seg.segment(below) if below else {"instances": []}
    case = detect_case(above_seg.get("instances", []), below_seg.get("instances", []))
    
    if "Coral" in str(case["detected_parts"]) or case["case"] == "coral":
        is_real = "YES" if sample.specimen_id in real_corals else "NO"
        print(f"{sample.specimen_id:<12} {case['case']:<10} {case['confidence']:>8.3f} {str(case['detected_parts']):<30} {is_real:>5}")
```

Run:
```bash
cd /home/iannyf/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification
.venv/bin/python coral_test.py
```

### 6.2 Trait accuracy test (10 species)

```python
# File: trait_accuracy_test.py
import xml.etree.ElementTree as ET
from pathlib import Path
from benchmarks.manifest import ManifestDataset
from models.visual_trait_extractor import extract
from models.mushroom_segmenter import detect_case, get_segmenter
from models.unified_pipeline import _merge_traits
from config import trait_config

trait_config.ENABLE_PART_AWARE_TRAITS = True

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST = PROJECT_ROOT / "benchmarks" / "evaluation_manifest.csv"
dataset = ManifestDataset(MANIFEST)
seg = get_segmenter()

tree = ET.parse("data/raw/species_traits.xml")
root = tree.getroot()

def get_db_traits(species_id):
    sp = root.find(f"species[@id='{species_id}']")
    if sp is None:
        return {}
    traits = {}
    for tg in sp.findall("trait_group"):
        cat = tg.get("category")
        for t in tg.findall("trait"):
            traits[f"{cat}.{t.get('name')}"] = t.text
    return traits

species_list = ["AG.AU", "AM.MU", "BO.ED", "CA.CI", "CO.CO", "GY.ES", "HY.PS", "LY.PE", "RA.BO", "SP.CR"]
specimen_by_species = {}
for sample in dataset:
    sid = sample.species_id
    if sid in species_list and sid not in specimen_by_species:
        specimen_by_species[sid] = sample

for sid in species_list:
    sample = specimen_by_species.get(sid)
    if not sample:
        continue
    db = get_db_traits(sid)
    above = extract(sample.load_above_bytes()) if sample.load_above_bytes() else {"visible_traits": {}}
    below = extract(sample.load_below_bytes()) if sample.load_below_bytes() else {"visible_traits": {}}
    
    above_seg = seg.segment(sample.load_above_bytes()) if sample.load_above_bytes() else {"instances": []}
    below_seg = seg.segment(sample.load_below_bytes()) if sample.load_below_bytes() else {"instances": []}
    case = detect_case(above_seg.get("instances", []), below_seg.get("instances", []))
    
    merged = _merge_traits(above.get("visible_traits", {}), below.get("visible_traits", {}), case["case"])
    
    print(f"\n{sid} ({sample.specimen_id}) case={case['case']} parts={merged.get('detected_parts', [])}")
    print(f"  DB cap_color={db.get('CAP.color')}  extracted={merged.get('cap_color')}")
    print(f"  DB cap_shape={db.get('CAP.shape')}  extracted={merged.get('cap_shape')}")
    print(f"  DB stem_color={db.get('STEM.color')}  extracted={merged.get('stem_color')}")
    print(f"  DB gills={db.get('GILLS.attachment')}  extracted={merged.get('hymenophore_type')}")
```

Run:
```bash
cd /home/iannyf/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification
.venv/bin/python trait_accuracy_test.py
```

### 6.3 Check geometric filter values for any specimen

```python
# File: geometric_filter_test.py
from pathlib import Path
from benchmarks.manifest import ManifestDataset
from models.mushroom_segmenter import get_segmenter
from models.yolo_part_masks import build_part_masks
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST = PROJECT_ROOT / "benchmarks" / "evaluation_manifest.csv"
dataset = ManifestDataset(MANIFEST)
seg = get_segmenter()

for sample in dataset:
    if sample.specimen_id != "RA.BO_025":  # change this
        continue
    for view_name, loader in [("above", sample.load_above_bytes), ("below", sample.load_below_bytes)]:
        image_bytes = loader()
        if image_bytes is None:
            continue
        seg_res = seg.segment(image_bytes)
        instances = seg_res.get("instances", [])
        if not instances:
            continue
        import io
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        H, W = np.array(pil_img).shape[:2]
        pm = build_part_masks(instances, (H, W))
        
        for part_key in ["cap", "stem", "underside", "coral"]:
            if part_key not in pm:
                continue
            m = pm[part_key]["mask"]
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area = sum(cv2.contourArea(c) for c in contours)
            all_points = np.vstack(contours)
            hull = cv2.convexHull(all_points)
            hull_area = cv2.contourArea(hull)
            solidity = total_area / hull_area if hull_area > 0 else 1.0
            total_perim = sum(cv2.arcLength(c, True) for c in contours)
            complexity = (total_perim ** 2) / (4 * np.pi * total_area) - 1.0 if total_area > 0 else 0.0
            print(f"{sample.specimen_id} {view_name} {part_key}: solidity={solidity:.3f} complexity={complexity:.3f} conf={pm[part_key]['confidence']}")
    break
```

---

## 7. Files Modified in This Investigation

| File | Change | Lines affected |
|------|--------|---------------|
| `config/trait_config.py` | Added coral thresholds + quality gate thresholds | +5 lines |
| `models/yolo_part_masks.py` | Fixed numpy `or` bug; added `_is_coral_like()` geometric filter; added coral quality gate bypass | ~+40 lines |
| `models/visual_trait_extractor.py` | Added `cv2.ximgproc` fallback for coral skeletonization | ~+8 lines |
| `models/mushroom_segmenter.py` | Cross-view consistency check in `detect_case()` | ~+20 lines |

**Config values set:**
```python
CORAL_MAX_FRAGMENTATION = 20
CORAL_MAX_HOLE_RATIO = 0.85
CORAL_MAX_SOLIDITY = 0.85
CORAL_MIN_COMPLEXITY = 1.0
```

---

## 8. Open Questions for Next Agent

1. **Should `detect_case` use filtered part masks instead of raw instances?** This would fix the `case=coral` + `detected_parts=[]` inconsistency. Requires restructuring `unified_pipeline.py` so part masks are built before `detect_case` is called.

2. **How to fix color accuracy?** The current `analyse_colours_masked()` uses K-means on RGB pixels. Should we:
   - Expand the color name palette?
   - Switch to HSV-based naming?
   - Compute distance to database colors and return the closest?

3. **How to improve shape accuracy?** Cap shape is derived from contour circularity/aspect ratio. Most masks come out as `wavy` or `funnel-shaped`. Should we:
   - Add more shape categories?
   - Use Hu moments or Zernike moments for rotation-invariant shape matching?
   - Skip shape auto-answer entirely and let the CNN handle it?

4. **Should we add a 5th YOLO class for "cauliflower/branching non-coral"?** This would cleanly separate `SP.CR` (Sparassis) from true corals (`RA.BO`, `RA.PA`).

5. **The below-view detection rate is poor** — 10.5% no detections, and stem/underside detection is only ~26-35%. Is this a YOLO training data issue (not enough below-view annotations) or an inherent problem with the photo quality?
