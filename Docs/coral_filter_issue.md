# Coral False-Positive Filter — Issue Documentation

> **Context:** The fine-tuned 4-class YOLO model (`data/Yolov8/best.pt`) occasionally predicts the `Coral` class on non-coral specimens. A geometric post-filter was implemented in `models/yolo_part_masks.py` to reject blob-like false positives while preserving real coral masks. This document records the findings and the unresolved trade-offs.

---

## 1. What was broken

When `ENABLE_PART_AWARE_TRAITS = True`, the trait extractor calls `build_part_masks()`, which groups YOLO instances by class. If YOLO predicts `Coral`, the code immediately sets:

```python
coarse_case = "coral"
```

**Example false positive:**
- `AG.AU_001` (above view) — YOLO outputs `1 Cap, 1 Coral` (coral conf = 0.321).
- The specimen is **not** a coral mushroom, yet the pipeline flagged it as `case=coral`.

## 2. Root cause of the false positive

The YOLO model itself is the source of the mislabel. The trait-extraction code was simply trusting the model output because the instance confidence (0.321) exceeded `MIN_CONFIDENCE = 0.30`.

Two bugs in the mask-handling code made the problem worse:
1. `inst.get("mask") or inst.get("cleaned_mask")` crashes on NumPy arrays (ambiguous truth value). The exception was silently swallowed, causing **every** image to fall back to whole-image analysis until the bug was fixed.
2. `cv2.ximgproc` is unavailable in the runtime, crashing coral skeleton analysis.

Both bugs were fixed before this investigation.

## 3. Geometric filter that was implemented

A `_is_coral_like(mask)` function was added to `models/yolo_part_masks.py`. It checks two shape metrics on the **merged** coral mask **after** it passes the quality gate:

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Solidity** | `contour_area / convex_hull_area` | `≤ 0.85` |
| **Complexity** | `(perimeter²) / (4π·area) − 1` | `≥ 1.0` |

**Rationale:** Real coral mushrooms have many gaps between branches → low solidity and/or very spiky outlines → high complexity. Classical mushroom false positives tend to be compact blobs → high solidity and low complexity.

The quality gate for coral was also relaxed (coral is naturally holey/fragmented):
- `CORAL_MAX_FRAGMENTATION = 20`
- `CORAL_MAX_HOLE_RATIO = 0.85`
- `MIN_CONFIDENCE = 0.30`

## 4. Benchmark dataset ground truth

The user confirmed that **only `RA.BO` and `RA.PA` specimens are real coral mushrooms** in the evaluation manifest.

| Specimen prefix | Species | Is coral? |
|-----------------|---------|-----------|
| `RA.BO_*` | *Ramaria botrytis* | **Yes** |
| `RA.PA_*` | *Ramaria pallida* | **Yes** |
| `AG.AU_*`, `HY.PS_*`, `LY.PE_*`, `SP.CR_*`, `BO.ED_*`, `LA.HE_*`, etc. | Various | **No** |

## 5. Measured geometric values

The table below shows the **merged** mask metrics for every image where YOLO predicted `Coral`.

| Specimen | View | YOLO Corals | Accepted by filter? | Solidity | Complexity | Defects | Notes |
|----------|------|------------|---------------------|----------|------------|---------|-------|
| **AG.AU_001** | above | 1 | **NO** | 0.901 | 0.884 | 46 | False positive — blob-like cap/stem clutter |
| **BO.ED_014** | above | 3 | NO | — | — | — | Rejected by quality gate (tiny area) |
| **HY.PS_019** | below | 2 | **NO** | 0.621* | 3.716 | 53 | False positive — *merged mask passes, but individual instances don't merge in `build_part_masks`; only the first instance is kept and it fails (solidity=0.885, complexity=1.497)* |
| **LA.HE_023** | below | 1 | NO | — | — | — | Rejected by quality gate |
| **LA.HE_024** | below | 1 | NO | — | — | — | Rejected by quality gate |
| **RA.BO_025** | below | 3 | **YES** | 0.818 | 4.130 | — | Real coral |
| **RA.BO_026** | above | 2 | **YES** | 0.650 | 6.644 | 66 | Real coral |
| **RA.BO_026** | below | 1 | **YES** | 0.564 | 13.050 | 78 | Real coral |
| **RA.BO_027** | above | 3 | **YES** | 0.806 | 3.202 | 31 | Real coral |
| **RA.BO_027** | below | 3 | **NO** | 0.855 | 2.459 | 19 | **Real coral REJECTED** — too compact/dense |
| **RA.PA_028** | above | 4 | **YES** | 0.760 | 3.049 | 37 | Real coral |
| **RA.PA_028** | below | 2 | **NO** | 0.873 | 1.286 | 25 | **Real coral REJECTED** — high solidity + low complexity; also merged hole_ratio=0.86 > 0.85 |
| **RA.PA_029** | above | 2 | **YES** | 0.798 | 3.094 | — | Real coral |
| **RA.PA_029** | below | 2 | **NO** | — | — | — | Rejected by quality gate or geometric filter |
| **RA.PA_030** | above | 3 | **NO** | 0.777 | 4.804 | 52 | Real coral — **rejected because instances don't all merge in `build_part_masks`; the kept cluster fails geometric** |
| **RA.PA_030** | below | 1 | **NO** | 0.921 | 2.022 | 28 | **Real coral REJECTED** — very compact single mask |
| **LY.PE_037** | above | 1 | **YES** | 0.593 | 10.446 | 120 | False positive — *mask is genuinely highly branching* |
| **LY.PE_037** | below | 2 | **YES** | 0.557 | 4.393 | 36 | False positive — *mask is genuinely highly branching* |
| **SP.CR_039** | above | 2 | **NO** | 0.867 | 1.755 | 24 | False positive — rejected correctly |
| **SP.CR_039** | below | 1 | **NO** | — | — | — | Rejected by quality gate |
| **PL.OS_045** | below | 1 | **NO** | — | — | — | Rejected by quality gate |
| **GA.MA_044** | below | 2 | **NO** | — | — | — | Rejected by quality gate |
| **HY.RE_050** | above | 1 | **NO** | — | — | — | Rejected by quality gate |
| **HY.RE_050** | below | 1 | **NO** | — | — | — | Rejected by quality gate |

## 6. Key findings

### 6.1 The primary false positive is fixed
`AG.AU_001` is now correctly rejected (solidity 0.901 > 0.85, complexity 0.884 < 1.0).

### 6.2 Real corals are being rejected too
Three real coral images fail the geometric filter:
- `RA.BO_027` below — dense coral cluster, merged solidity = 0.855
- `RA.PA_028` below — dense coral cluster, merged solidity = 0.873
- `RA.PA_030` below — single very large compact coral mask, solidity = 0.921

These masks are **genuinely compact** because the coral branches are thick and fill most of the convex hull.

### 6.3 Some false positives look genuinely branching
`LY.PE_037` (above & below) is **not** a coral species, yet its YOLO mask has:
- Very low solidity (0.593 / 0.557)
- Very high complexity (10.446 / 4.393)
- Very high convexity-defect count (120 / 36)

Geometrically, this mask looks **more** like coral than some real corals do. The error originates in the YOLO model, not the filter.

### 6.4 There is no clean threshold separation
Plotting the merged masks on a solidity-vs-complexity plane shows heavy overlap between real corals and false positives:

| | False positives | Real corals |
|---|---|---|
| High solidity (> 0.85), low complexity (< 2.0) | AG.AU_001 | RA.BO_027 below, RA.PA_028 below, RA.PA_030 below |
| Low solidity (< 0.85), high complexity (> 3.0) | LY.PE_037 | RA.BO_026, RA.BO_027 above |

**There is no single pair of thresholds that rejects all false positives while accepting all real corals.**

### 6.5 `build_part_masks` discards non-merging instances
The merging logic only keeps the first cluster (sorted by confidence). For `RA.PA_030` above, three coral instances are detected but only some merge; the rest are thrown away. This means even if a smaller branch passes the geometric filter, it is lost.

## 7. Attempted fixes and why they fail

| Idea | Why it fails |
|------|-------------|
| Raise `CORAL_MAX_SOLIDITY` to 0.95 | Would allow the three rejected real corals, but also accepts `HY.PS_019` below (false positive, solidity = 0.885). |
| Raise `CORAL_MIN_COMPLEXITY` to 1.5 | Rejects `HY.PS_019` (1.497), but also rejects `RA.PA_028` below (1.286) and `RA.PA_030` below (2.022). |
| Use convexity-defect count instead | `AG.AU_001` has 46 defects (high!) yet is a false positive. Defect count correlates with mask size, not coral-ness. |
| Require `area_ratio > 0.25` for high-solidity masks | Would accept `RA.PA_030` below (area = 0.80) and reject `HY.PS_019` below (area = 0.07). But `RA.PA_030` above (area = 0.31) and `RA.PA_028` above instances (areas ~0.04) would be rejected. |
| Skip geometric filter when multiple coral instances merge | `RA.BO_027` below and `RA.PA_028` below would pass, but `LY.PE_037` below (2 instances) would also pass. `RA.PA_030` below (1 instance) would still fail. |

## 8. Open questions for further work

1. **Should the filter optimize for precision or recall?**
   - Precision-first (current): `AG.AU_001` is rejected, but 3 real corals are lost.
   - Recall-first: All real corals pass, but `HY.PS_019` below and possibly `LY.PE_037` are accepted.

2. **Is the issue better fixed in the YOLO model?**
   The false positives (`AG.AU_001`, `HY.PS_019`, `LY.PE_037`) are misclassified by the model itself. Retraining with more negative coral examples might be more effective than post-hoc geometric filtering.

3. **Should coral detection use multi-image context?**
   A real coral specimen (`RA.BO_*` / `RA.PA_*`) usually has coral detected in **both** above and below views. False positives often appear in only one view. Cross-photo consistency could help.

4. **Should `build_part_masks` keep multiple disjoint masks per part?**
   Currently only the first cluster is kept. For branching fungi, keeping all clusters and analysing them together (or selecting the most geometrically coral-like one) could improve recall.

## 9. Files touched

- `models/yolo_part_masks.py` — added `_is_coral_like()` and relaxed `_passes_quality_gate()` for coral
- `config/trait_config.py` — added `CORAL_MAX_FRAGMENTATION`, `CORAL_MAX_HOLE_RATIO`, `CORAL_MAX_SOLIDITY`, `CORAL_MIN_COMPLEXITY`
- `models/visual_trait_extractor.py` — added `cv2.ximgproc` fallback for skeletonization

## 10. How to reproduce

```bash
cd /home/iannyf/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification
.venv/bin/python -c "
from benchmarks.manifest import ManifestDataset
from models.mushroom_segmenter import get_segmenter
from models.yolo_part_masks import build_part_masks
# ... see diagnostic scripts in this doc for full reproduction
"
```

The benchmark manifest is at `benchmarks/evaluation_manifest.csv` (23 species, 57 specimens, 114 images). YOLO weights are at `data/Yolov8/best.pt`.
