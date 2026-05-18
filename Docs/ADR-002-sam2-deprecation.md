# ADR-002: Deprecation of SAM2 in Favor of Roboflow Manual Annotation

**Status:** Accepted  
**Date:** 2026-05-12  
**Context:** YOLOv8 segmentation training data pipeline

## Problem

The project initially used SAM2 (Segment Anything Model 2) to generate ~350 auto-masks as pseudo-labels for YOLOv8 training. This was intended as a fast way to bootstrap a segmentation dataset without manual annotation.

## Decision

**SAM2 and all derived artifacts are deprecated.** The project now uses **Roboflow manual annotation** as the sole source of training data.

## Rationale

### 1. SAM2 is single-class
SAM2 produces binary masks (`mushroom` vs `background`). It cannot distinguish:
- `Cap` vs `Stem` vs `Underside` vs `Coral`

Converting SAM2 masks into 4-class training data requires the same manual tracing effort as annotating from scratch in Roboflow.

### 2. Zero image overlap
| Dataset | Images | Source | Overlap with Roboflow |
|---------|--------|--------|----------------------|
| SAM2 masks | 352 | Auto-generated from iNaturalist downloads | **0** |
| Roboflow annotations | 139 | Manually annotated, curated | 139 |

The SAM2 masks are from a completely different image set. They cannot augment the current training data without re-annotating parts.

### 3. Bottleneck is part-classification, not detection
The current YOLOv8 model detects mushrooms well. The weakness is **Underside** (low mAP), caused by:
- Visual similarity between Cap and Underside
- Fewer underside-only training examples
- Pixel-level semantic confusion

SAM2 does not solve semantic part confusion — only more part-specific labeled data does.

### 4. Historical value preserved
The SAM2 experiment is documented in:
- `Docs/TUTORIAL_SAM2_YOLOV8.md` — explains the SAM2→YOLOv8 concept
- `scripts/evaluate_segmentation.py` — segmentation evaluation logic (reused)

These files are kept for thesis context. The actual SAM2 scripts and mask data are removed.

## Consequences

- **Positive:** Cleaner repo, no confusion about which data source is authoritative.
- **Positive:** ~350 MB of mask data freed.
- **Neutral:** Future retraining uses Roboflow exports directly (`data.yaml` → Colab).
- **Negative:** None. SAM2 was already unused in the active pipeline.

## Removed Artifacts

```
data/SegMaskSAM2/                          # 352 binary PNG masks
data/segmentation/                         # Old SAM2-derived YOLO dataset (312 imgs, 1 class)
scripts/sam2_pilot.py                      # SAM2 quality gate script
scripts/generate_sam2_masks.py             # Batch SAM2 mask generation
scripts/generate_sam2_eval_masks.py        # SAM2 eval mask generation
scripts/retry_failed_sam2.py               # Retry logic for failed SAM2 runs
scripts/prepare_yolo_seg_dataset.py        # SAM2 masks → YOLO converter
```

## Updated Artifacts

```
scripts/train_yolov8_seg.py                # Metadata label changed: "sam2-v1" → "roboflow-4class"
```
