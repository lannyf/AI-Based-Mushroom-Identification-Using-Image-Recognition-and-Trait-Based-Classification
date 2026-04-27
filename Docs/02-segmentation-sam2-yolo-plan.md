# Implementation Plan 1: YOLOv8-seg Fine-Tuning for Mushroom Segmentation

## 1. Why Fine-Tuning Is Needed

The current pipeline uses a generic COCO-pretrained `yolov8n-seg.pt` model. COCO contains 80 classes (person, car, dog, etc.) but **no mushroom class**. When the model sees a mushroom photo, it segments whatever generic object has the strongest visual saliency — often a hand holding the mushroom, a bright leaf, or a blurred background object. The quality-gating heuristics (confidence, area ratio, fragmentation) can reject some bad masks, but they cannot fix a model that was never trained on the target domain.

**What fine-tuning fixes:**
- Teaches the model the specific visual features of mushrooms: cap contours, stem silhouettes, colour patterns, and typical poses.
- Dramatically reduces false detections of hands, leaves, and forest floor.
- Keeps the model tiny (`yolov8n-seg` is ~6 MB) and fast enough for CPU inference in a FastAPI backend or mobile deployment.
- Produces deterministic, reproducible masks at runtime without relying on heuristic post-processing to save bad predictions.

## 2. How Fine-Tuning Improves the Problem

| Current State (Generic YOLOv8-seg) | After Fine-Tuning |
|---|---|
| Segments hands, background, leaves | Segments mushrooms consistently |
| High false-positive rate for non-mushroom objects | Single-class "mushroom" detector |
| Quality gating rejects ~30-50% of masks | Quality gating accepts >90% of masks |
| Trait extraction often falls back to full-image | Masked trait extraction becomes the default path |
| Unpredictable behaviour on unseen mushroom angles | Learns domain-specific features from training data |

The downstream effect is that `analyse_colours_masked`, `analyse_shape_masked`, `analyse_texture_masked`, and `analyse_brightness_masked` in `visual_trait_extractor.py` receive clean mushroom-only pixels. This eliminates background colour contamination, false ridge detection from grass, and shape distortion from bright background objects.

## 3. Implementation Steps

### 3.1 Prerequisite: Labeled Masks
**CRITICAL GAP / DECISION POINT:**
Fine-tuning requires polygon/mask annotations. You have two paths:

| Path | Effort | Quality | Recommendation |
|---|---|---|---|
| A. Use SAM 2 generated masks (Plan 2) | Low (~30 min review) | Very high | **Recommended** |
| B. Manual annotation with Label Studio / CVAT | High (~6–10 hours) | High | Only if SAM 2 fails |

**This plan assumes Path A.** If you skip Plan 2, you must manually annotate ~280 training masks first.

### 3.2 Data Preparation
YOLOv8-seg expects a specific dataset layout:
```
data/segmentation/
  ├── dataset.yaml
  ├── images/
  │     ├── train/
  │     └── val/
  └── labels/
        ├── train/
        └── val/
```
Each label file is a `.txt` containing one line per instance with normalized polygon vertices.

**Deliverable:** `scripts/prepare_yolo_seg_dataset.py`
- Accepts a directory of images + binary masks (from SAM 2 or manual annotation).
- Converts masks to YOLO polygon format using OpenCV `findContours` + Ramer-Douglas-Peucker simplification.
- Handles edge cases: multi-component masks (keep largest contour), holes (ignore inner contours for YOLO format), tiny masks (< 50 pixels → skip).
- Single class: `0 = mushroom`.
- Performs an 80/20 train/validation split, stratified by species folder if possible.
- Writes `data/segmentation/dataset.yaml` with absolute or relative paths and class names.
- **Validation step:** Script should print a summary (train images, val images, average polygon points) and warn if any label file is empty.

### 3.3 Training Pipeline
**Deliverable:** `scripts/train_yolov8_seg.py`
```python
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model.train(
    data="data/segmentation/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    name="mushroom_seg",
    project="artifacts/yolov8_seg_runs",
    exist_ok=True,
    patience=20,
    close_mosaic=10,
)
```
- Start from pretrained COCO weights (transfer learning).
- Image size: 640.
- Early stopping patience: 20 epochs.
- Disable mosaic augmentation for the final 10 epochs to stabilize mask boundaries.
- **Metrics to watch during training:** `seg/mAP50`, `seg/mAP50-95`. Target: `seg/mAP50 > 0.85` on validation.

**Hardware note:** No GPU is available on this machine. Training 100 epochs of yolov8n-seg on ~280 images on a consumer CPU will take **10–24 hours** (not 4–12). The **primary recommended path is Google Colab**:
1. Upload `data/segmentation/` to Google Drive.
2. Mount Drive in Colab, run `train_yolov8_seg.py`.
3. Training time on free T4 GPU: ~15–30 minutes.
4. Download `best.pt` to `artifacts/yolov8_seg_ft.pt`.

### 3.4 Model Deployment & Rollback Strategy
- Copy best checkpoint to `artifacts/yolov8_seg_ft.pt`.
- Write metadata to `artifacts/yolov8_seg_ft_metadata.json`:
  ```json
  {
    "model": "yolov8n-seg",
    "training_date": "2026-04-17",
    "dataset_version": "sam2-v1",
    "train_images": 280,
    "val_images": 72,
    "epochs": 100,
    "imgsz": 640,
    "class_map": {"0": "mushroom"},
    "source_checkpoint": "yolov8n-seg.pt"
  }
  ```
- Update `models/mushroom_segmenter.py` to accept a configurable model path (default: `artifacts/yolov8_seg_ft.pt`).
- Keep generic `yolov8n-seg.pt` as an explicit fallback if the fine-tuned model file is missing.
- **A/B testing:** Before promoting the fine-tuned model as default, run `scripts/evaluate_segmentation.py` (see Testing) on both generic and fine-tuned models. Only switch the default if fine-tuned IoU is higher.

### 3.5 Post-Processing Improvements
Update `models/mushroom_segmenter.py` with smarter selection heuristics:
1. **Center-bias tiebreaker**: Prefer masks whose bbox centroid is closest to the image center.
2. **Skin-colour rejection**: Reject masks where >30% of pixels fall in HSV skin-tone range (H: 0–50, S: 20–170, V: 50–255).
3. **Aspect ratio guard**: Reject extreme aspect ratios (`w/h > 4` or `< 0.25`).

Tune `config/segmentation_config.py` thresholds against validation outputs. Expect `MIN_MASK_CONFIDENCE` to increase to 0.60–0.70 because the fine-tuned model is more confident on actual mushrooms.

## 4. Testing Strategy

### 4.1 Unit Tests
**Deliverable:** `tests/test_mushroom_segmenter.py`
- Mock YOLO result parsing when `ultralytics` is unavailable.
- Verify mask cleanup (small fragment removal, morphological operations).
- Verify quality metrics (area_ratio, fragmentation, hole_ratio, boundary_irregularity).
- Verify deterministic selection logic (confidence sorting, center-bias tiebreaker, skin-rejection).
- Verify fallback to generic model when `artifacts/yolov8_seg_ft.pt` is missing.

### 4.2 Regression Tests
Run the full existing test suite:
```bash
pytest tests/ -v
```
**Acceptance criteria:**
- All 160 existing tests must pass.
- `test_trait_regression_real_images.py`: Real images must still produce correct dominant colours and correct tree traversal conclusions.
- `test_visual_trait_extractor.py`: Synthetic colour/shape/texture tests must pass.

### 4.3 Segmentation-Specific Evaluation
**Deliverable:** `scripts/evaluate_segmentation.py`
If a held-out manually-annotated mask set exists (20–30 images):
- Compute **Mask IoU**, **Precision**, **Recall** at IoU thresholds 0.50 and 0.75.

If no ground-truth exists, compute proxy metrics:
- **Trait stability**: Run trait extraction on 50 images with old vs new segmenter. Measure dominant_colour / cap_shape change rate.
- **Fallback rate**: Percentage of images where quality gating falls back to full-image traits.
- **Wrong-object rate**: Manual visual inspection of 50 masks, counting hands/background segmented instead of mushroom. Target: < 5%.
- **Speed**: Mean and p95 inference time per image on CPU. Target: < 300ms.

### 4.4 Promotion Gate
Before making `artifacts/yolov8_seg_ft.pt` the default, all of the following must be true:
1. `evaluate_segmentation.py` shows improvement over generic `yolov8n-seg` on at least 2 of 3 metrics (IoU, fallback rate, wrong-object rate).
2. Full `pytest tests/` passes with 0 failures.
3. CPU inference time is within 150% of the generic model (i.e., not dramatically slower).

## 5. Risks, Mitigations & Readiness Assessment

| Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|
| 352 images too few for robust fine-tuning | Medium | High | Transfer learning + heavy augmentation. Validate against SAM 2 masks. | Mitigable |
| CPU training takes 10–24 hours | High | Medium | Use Google Colab free GPU (primary path). | Mitigable |
| Overfitting to limited angles/species | Medium | High | Ensure train split includes varied poses/lighting. Use early stopping. | Mitigable |
| Integration breaks existing tests | Low | Medium | Run full regression suite after each change. Keep full-image fallback. | Mitigable |
| No labeled masks available if Plan 2 skipped | High | Blocking | Plan 2 (SAM 2) is a prerequisite. Manual annotation is the fallback. | **Requires decision** |
| Ultralytics version compatibility | Low | Medium | Pin `ultralytics>=8.0.0` in requirements.txt. | Mitigable |

**Readiness verdict:** Ready for implementation **only if** Plan 2 (SAM 2 mask generation) is executed first OR a manual annotation effort is undertaken. The training script, dataset preparation, and evaluation logic can be written immediately, but training cannot begin without labeled masks.

---

# Implementation Plan 2: SAM 2 Integration for Mushroom Segmentation

## 1. Why SAM 2 Is Needed

Even after fine-tuning YOLOv8-seg, two problems remain:

1. **Data bottleneck for fine-tuning**: To train YOLOv8-seg, you need high-quality masks. Drawing 352 polygon masks by hand is tedious and error-prone.
2. **Zero-shot accuracy on hard cases**: A fine-tuned YOLOv8-seg model trained on 352 images may still struggle with unusual poses, heavy occlusion, or species not well-represented in the training set. SAM 2 was designed for exactly these zero-shot segmentation scenarios — it can segment almost any object given a simple point or box prompt, without ever having seen that specific object class during training.

**What SAM 2 fixes:**
- Generates near-perfect masks automatically using only a point prompt (e.g., "the object at the center of the image").
- Eliminates the need to hand-draw hundreds of polygon annotations.
- Serves as a **ground-truth generator** for training data and as a **quality benchmark** to evaluate the fine-tuned YOLOv8-seg model.
- Can be kept as a high-accuracy **offline fallback** for images where the runtime segmenter fails.

## 2. How SAM 2 Improves the Problem

| Problem | SAM 2 Solution |
|---|---|
| Need 352 manual mask annotations for YOLO training | SAM 2 generates ~90-95% of masks automatically; only correct the failures |
| Generic YOLO segments wrong objects | SAM 2 with a center point prompt segments exactly the intended object |
| No way to verify if fine-tuned YOLO masks are accurate | Compare YOLO masks against SAM 2 masks on the same images (IoU benchmark) |
| Hard cases (occlusion, unusual angles) fail YOLO | SAM 2 can be re-prompted with multiple positive/negative points |

SAM 2 does **not** replace YOLOv8-seg at runtime because it is too large and slow for real-time API use. Its role is **offline data generation and validation**.

## 3. Implementation Steps

### 3.1 Dependency Check & Installation
**CRITICAL: Verify environment before implementation.**
- SAM 2 requires PyTorch. Check: `python -c "import torch; print(torch.__version__)"`
- If PyTorch is missing from the active environment, install it first: `pip install torch torchvision`.
- Install SAM 2: `pip install git+https://github.com/facebookresearch/sam2.git` (official repo). Note: a simple `pip install sam2` on PyPI may not point to the official Meta implementation; use the GitHub URL.
- Download checkpoint to `artifacts/sam2_hiera_tiny.pt` (~35 MB) or `artifacts/sam2_hiera_small.pt` (~80 MB).
- **Disk space check:** Ensure ~500 MB free for checkpoints + generated masks.
- Add to `requirements.txt`:
  ```
  torch>=2.0.0
  torchvision>=0.15.0
  # SAM 2 installed from GitHub; not on PyPI as of 2026-04
  # pip install git+https://github.com/facebookresearch/sam2.git
  ```

### 3.2 Prompt Strategy
A naive "center point" prompt fails when mushrooms are off-center (common in field photos).

**Recommended multi-strategy approach:**
1. **Primary strategy: Center point + 4 corner negative points**
   - Positive point: image center `(cx, cy)`.
   - Negative points: four corners of the image to discourage background segmentation.
   - This tells SAM 2: "segment the object near the center, not the background at the edges."
2. **Fallback strategy: Bounding box prompt**
   - Use the existing generic YOLOv8-seg bbox as a loose box prompt for SAM 2.
   - This helps when the mushroom is off-center but YOLO still detects something vaguely mushroom-shaped.
3. **Correction strategy: Multi-point prompting**
   - If the first mask is poor, add positive points on the mushroom cap and negative points on intruding hands/leaves.

### 3.3 Mask Generation Script
**Deliverable:** `scripts/generate_sam2_masks.py`
- Load SAM 2 model (`SAM2ImagePredictor` or equivalent API).
- For each image in `data/raw/images/`:
  1. Load image and compute center point.
  2. **Attempt 1:** Pass center positive point + 4 corner negative points.
  3. Receive 3 candidate masks from SAM 2. Select the one with highest predicted IoU score.
  4. If score < 0.85, **Attempt 2:** Use the generic YOLO bbox as a box prompt.
  5. If still poor, **Attempt 3:** Log image for manual review.
  6. Save best mask as PNG to `data/SegMaskSAM2/` (binary uint8, 0/255, same dimensions as input).
  7. Save metadata to `data/SegMaskSAM2/manifest.json`:
     ```json
     {
       "image": "data/raw/images/AM.MU/Amanita_muscaria_1.jpg",
       "mask_path": "data/SegMaskSAM2/00e10b3b...png",
       "prompt_type": "point+negative",
       "sam_score": 0.94,
       "attempt": 1
     }
     ```
- **Processing note:** SAM 2 image encoder on CPU takes ~3–8 seconds per image. For 352 images, expect **25–50 minutes total**. Run as a background batch job.
- **Resume support:** Skip images already present in `data/SegMaskSAM2/` to allow resuming interrupted runs.

### 3.4 Quality Control & Correction Workflow
- Batch-review all generated masks visually (fastest way: generate an HTML gallery or use a local image viewer).
- **Expected accuracy:**
  - Centered, single-mushroom photos: ~95% perfect.
  - Multi-mushroom photos: ~70% (may segment wrong mushroom).
  - Heavy hand occlusion: ~60% (may include hand).
- **Correction options for failures:**
  1. **Re-prompt SAM 2**: Add manual negative points to exclude hands/background. Add positive points on the correct cap.
  2. **Manual polygon correction**: Use Label Studio or CVAT to edit the mask boundary.
  3. **Discard**: If the image is unusable (extreme blur, mushroom invisible), exclude from training data.
- Target: < 30 images require any correction. < 10 images require full manual redraw.

### 3.5 Integration with Plan 1
SAM 2 is the **data source** for Plan 1:
- After QC, run `scripts/prepare_yolo_seg_dataset.py` using `data/SegMaskSAM2/` as the mask source.
- This closes the loop: SAM 2 provides the annotations that enable YOLOv8-seg fine-tuning.

### 3.6 Benchmark Script (Optional but Recommended)
**Deliverable:** `scripts/benchmark_sam2_vs_yolo.py`
- Run both SAM 2 (best prompt) and fine-tuned YOLOv8-seg on a 50-image held-out subset.
- Compute mask IoU between the two systems.
- Where IoU is low (< 0.80), visually inspect to determine which mask is correct.
- Output a CSV report: `image, sam2_iou, yolo_iou, winner, notes`.
- Goal: ensure fine-tuned YOLOv8-seg masks are within ~5% IoU of SAM 2 masks on average.

## 4. Testing Strategy

### 4.1 Smoke Test
**Deliverable:** `tests/test_sam2_generator.py`
- Verify `scripts/generate_sam2_masks.py` runs without errors on a single test image.
- Verify output mask file exists, has correct dimensions matching input image, and contains only 0/255 values.
- Verify manifest JSON is valid and contains required keys (`image`, `mask_path`, `prompt_type`, `sam_score`).
- Verify resume behaviour: skip already-processed images.

### 4.2 Mask Quality Evaluation
- Randomly sample 50 generated masks.
- Manual visual inspection by a human: classify each mask as:
  - **Correct**: Mushroom boundary is accurate, no major background inclusion.
  - **Partially correct**: Minor errors (small inclusion/exclusion) but usable for training.
  - **Wrong**: Major errors (wrong object, heavy hand inclusion, missing most of mushroom).
- **Targets:**
  - Correct: > 85%
  - Partially correct: < 10%
  - Wrong: < 5%
- If wrong rate exceeds 5%, adjust prompting strategy before proceeding to Plan 1.

### 4.3 Trait Regression Test
- Temporarily modify `models/mushroom_segmenter.py` to load SAM 2 masks from `data/SegMaskSAM2/` instead of running YOLO inference.
- Run `test_trait_regression_real_images.py`.
- Verify that:
  - Fly agaric images still produce red-dominant traits.
  - Black trumpet images still produce dark-dominant traits.
  - Bolete images do not falsely trigger ridge detection.
  - All tree traversal tests still reach correct conclusions.
- This validates that SAM 2 masks are good enough to produce reliable traits.

### 4.4 Performance Test
- Measure SAM 2 inference time per image on CPU with `sam2_hiera_tiny`.
- Expected: 3–8 seconds per image (image encoder + mask decoder).
- Document this in a README note: "SAM 2 is an offline batch tool, not suitable for real-time API inference."

## 5. Risks, Mitigations & Readiness Assessment

| Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|
| SAM 2 GitHub install fails or requires specific CUDA version | Medium | Blocking | Install CPU-only PyTorch first, then SAM 2 from GitHub. Test in a fresh venv. | Mitigable |
| SAM 2 image encoder uses too much RAM on CPU | Medium | High | Use `sam2_hiera_tiny`. Process one image at a time. Call `torch.cuda.empty_cache()` if applicable. | Mitigable |
| Center-point prompt fails on off-center mushrooms | Medium | Medium | Use negative corner points + YOLO bbox fallback prompt. | Mitigable |
| SAM 2 includes hands in mask | Medium | Medium | Add negative points on hand regions, or correct manually. | Mitigable |
| Processing 352 images takes >1 hour on CPU | High | Low | Run overnight. Results are cached and resumable. | Acceptable |
| SAM 2 masks worse than expected, requiring heavy manual correction | Low | High | If >20% need correction, switch to manual annotation for the worst images and use SAM 2 for the easy ones. | Mitigable |

**Readiness verdict:** Ready for implementation. The primary risk is installation (PyTorch + SAM 2 from GitHub), which should be tested in an isolated environment first. No labeled data prerequisites — SAM 2 is the prerequisite for Plan 1.

---

# Summary: Execution Order & Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│  PLAN 2: SAM 2 (Offline Data Generation) — NO PREREQUISITES │
│  ├─ Install torch + SAM 2 from GitHub                       │
│  ├─ Generate masks for 352 images (~30-50 min CPU batch)    │
│  ├─ Quality control: review & correct ~10-30 failures       │
│  └─ Outputs: data/SegMaskSAM2/ + manifest.json              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ masks become training labels
┌─────────────────────────────────────────────────────────────┐
│  PLAN 1: YOLOv8-seg Fine-Tuning — REQUIRES Plan 2 outputs   │
│  ├─ Prepare YOLO dataset from SAM 2 masks                   │
│  ├─ Train on Google Colab (~15-30 min GPU)                  │
│  ├─ Evaluate vs. generic model                              │
│  ├─ Promote if metrics improve                              │
│  └─ Deploy as fast runtime segmenter                        │
└─────────────────────────────────────────────────────────────┘
```

**Recommended execution order:**
1. Implement Plan 2 first (SAM 2 mask generation).
2. Run quality control on SAM 2 outputs.
3. Implement Plan 1 dataset preparation + training script.
4. Train YOLOv8-seg (preferably on Colab).
5. Evaluate and promote the fine-tuned model.
