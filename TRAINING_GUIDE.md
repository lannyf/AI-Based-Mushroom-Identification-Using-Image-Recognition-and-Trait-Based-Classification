# Model Retraining Guide

Complete step-by-step guide for retraining the CNN classifier, YOLOv8 segmentation model, and regenerating SAM 2 masks. Designed so you (or anyone else) can reproduce the entire training pipeline from scratch.

---

## Table of Contents

1. [What You Need](#1-what-you-need)
2. [Understanding Your Data](#2-understanding-your-data)
3. [Step 1 — CNN Classifier (Local CPU)](#step-1--cnn-classifier-local-cpu)
4. [Step 2 — SAM 2 Mask Generation (Local CPU)](#step-2--sam-2-mask-generation-local-cpu)
5. [Step 3 — YOLOv8 Segmentation (Google Colab)](#step-3--yolov8-segmentation-google-colab)
6. [Step 4 — Evaluation & Promotion](#step-4--evaluation--promotion)
7. [Troubleshooting](#troubleshooting)

---

## 1. What You Need

| Requirement | CNN | SAM 2 | YOLOv8 |
|-------------|-----|-------|--------|
| **Hardware** | Local CPU (AMD Ryzen 7 5825U) | Local CPU | Google Colab T4 GPU |
| **Time** | ~45–90 min | ~15–25 min | ~15–30 min |
| **Python env** | `.venv` with `torch`, `timm` | `.venv` with `torch`, `sam2` | Colab (preinstalled) |
| **Data** | `data/raw/images/` (7 species) | `data/raw/images/` + `data/raw/evaluation_images/` | `data/segmentation/` (prepared dataset) |

**Activate your local environment:**
```bash
cd /path/to/project
source .venv/bin/activate
```

---

## 2. Understanding Your Data

Before training, know what each folder contains:

| Folder | Contents | Used By |
|--------|----------|---------|
| `data/raw/images/AM.MU/` | 30 *Amanita muscaria* images | CNN, YOLO, SAM 2 |
| `data/raw/images/AM.VI/` | 30 *Amanita virosa* images | CNN, YOLO, SAM 2 |
| `data/raw/images/BO.ED/` | 30 *Boletus edulis* images | CNN, YOLO, SAM 2 |
| `data/raw/images/BO.BA/` | 30 *Boletus badius* images | CNN, YOLO, SAM 2 |
| `data/raw/images/CA.CI/` | 30 *Cantharellus cibarius* images | CNN, YOLO, SAM 2 |
| `data/raw/images/CR.CO/` | 30 *Craterellus cornucopioides* images | CNN, YOLO, SAM 2 |
| `data/raw/images/HY.PS/` | 30 *Hygrophoropsis aurantiaca* images | CNN, YOLO, SAM 2 |
| `data/raw/images/coprinus_comatus/` | 30 images | YOLO, SAM 2 only |
| `data/raw/images/fomitopsis_betulina/` | 30 images | YOLO, SAM 2 only |
| `data/raw/images/lycoperdon_utriforme/` | 22 images | YOLO, SAM 2 only |
| `data/raw/images/ramaria_botrytis/` | 30 images | YOLO, SAM 2 only |
| `data/raw/images/sparassis_crispa/` | 30 images | YOLO, SAM 2 only |
| `data/raw/background/` | 48 no-mushroom images | YOLO training only |
| `data/raw/evaluation_images/` | 60 holdout test images | SAM 2 evaluation masks |
| `data/Mushroom segmentation.coco-segmentation/` | 12 manually annotated masks | YOLO training (optional) or evaluation ground truth |
| `data/SegMaskSAM2/` | 352 auto-generated SAM 2 masks | YOLO training labels |

**Important:** The CNN is configured for **7 species only** (`config/image_model_config.py`). YOLOv8 and SAM 2 use **all 13 species folders**.

---

## Step 1 — CNN Classifier (Local CPU)

### What It Does
Trains an EfficientNet-B3 classifier to recognize 7 mushroom species from photos. Uses transfer learning from ImageNet.

### Configuration
Edit `config/image_model_config.py` if you want to change defaults:
```python
BASE_MODEL = "efficientnet_b3"      # Don't change without good reason
DEFAULT_EPOCHS = 20                  # Usually enough with early stopping
DEFAULT_BATCH_SIZE = 8               # Fits in 13 GB RAM
DEFAULT_LEARNING_RATE = 3e-4         # Standard for transfer learning
NUM_CLASSES = 7                      # Fixed: Fly Agaric, Chanterelle, False Chanterelle, Porcini, Other Boletus, Amanita virosa, Black Trumpet
```

### Run Training
```bash
cd /path/to/project
source .venv/bin/activate

# Default settings (20 epochs, batch 8, lr 3e-4)
python scripts/train_cnn.py

# Or customize:
python scripts/train_cnn.py --epochs 30 --batch-size 8 --lr 3e-4
```

### What Happens
1. **Phase 1:** Trains only the classification head (frozen backbone) for ~7 epochs
2. **Phase 2:** Unfreezes the full network and fine-tunes at reduced LR (~0.1×)
3. **Early stopping:** If validation accuracy doesn't improve for 5 epochs, training stops
4. **Output:** Best checkpoint saved to `artifacts/cnn_weights.pt`, history to `artifacts/cnn_training_history.json`

### Expected Results
- Time: **45–90 minutes** on your AMD Ryzen 7 5825U
- Final val accuracy: ~75–85% (varies by random seed and split)
- Training loss should drop from ~2.2 to <0.1

### Verify It Worked
```bash
python -c "import json; h=json.load(open('artifacts/cnn_training_history.json')); print(f'Epochs: {len(h[\"val_acc\"])}, Best val_acc: {max(h[\"val_acc\"]):.3f}')"
```

---

## Step 2 — SAM 2 Mask Generation (Local CPU)

### What It Does
Generates binary segmentation masks for every training and evaluation image using Meta's SAM 2 in zero-shot mode. **You do not train SAM 2.** You run it as a pretrained model.

### Prerequisites
SAM 2 must be installed in your `.venv` and the checkpoint downloaded:
```bash
ls artifacts/sam2.1_hiera_tiny.pt   # Should exist (156 MB)
```

If missing, see `sam2/INSTALL.md` or run:
```bash
pip install -e ./sam2
wget -O artifacts/sam2.1_hiera_tiny.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
```

### Generate Training Masks (352 images)
```bash
cd /tmp   # MUST run from outside the sam2/ repo folder
python /path/to/project/scripts/generate_sam2_masks.py \
  --project-root /path/to/project \
  --images-dir data/raw/images \
  --output-dir data/SegMaskSAM2
```

**Strategy:**
- Attempt 1: Center point + 4 corner negative points
- Attempt 2 (fallback): Generic YOLOv8 bbox as box prompt
- Ranks 3 candidate masks by prompt overlap, compactness, and border touch

**Time:** ~15–25 minutes for all 352 images.

### Generate Evaluation Masks (60 images)
```bash
cd /tmp

# Holdout set (30 images)
python /path/to/project/scripts/generate_sam2_eval_masks.py \
  --project-root /path/to/project \
  --eval-list data/raw/eval_holdout_30.txt \
  --output-dir data/SegMaskSAM2_eval_holdout

# Secondary set (30 images)
python /path/to/project/scripts/generate_sam2_eval_masks.py \
  --project-root /path/to/project \
  --eval-list data/raw/eval_secondary_30.txt \
  --output-dir data/SegMaskSAM2_eval_secondary
```

### Output Files
- `data/SegMaskSAM2/<image_name>_sam2.png` — binary mask (0 or 255)
- `data/SegMaskSAM2/manifest.jsonl` — per-image metadata (strategy, scores, timing)
- `data/SegMaskSAM2/summary.json` — aggregate statistics

---

## Step 2b — Manual Annotation with Roboflow (Optional but Recommended)

If you want higher-quality masks than SAM 2 alone, annotate a subset of images manually. **Roboflow's AI-assisted polygon tool** reduces annotation time from ~5 minutes per image to ~10 seconds per image.

### Why Bother?
- SAM 2 makes mistakes on hands, shadows, and off-center mushrooms
- ~40 high-quality manual annotations can improve mean IoU from ~0.45 to ~0.65+
- Gives you honest **evaluation ground truth** (human-verified, not AI-generated)

### Annotation Rules
| Include in Mask | Exclude from Mask |
|-----------------|-------------------|
| Mushroom cap | Human hands |
| Mushroom stem/stipe | Forest floor / soil |
| Gills / pore surface (if visible) | Grass, leaves, twigs |
| Partial mushroom (if occluded) | Occluding objects |

**Do NOT crop images.** Keep full frames so the model learns scale and context.

### Roboflow Workflow

1. **Create project**
   - Go to https://roboflow.com
   - Create new project → Instance Segmentation
   - Class: `mushroom` (single class)

2. **Upload images**
   - Upload ~50 images you want to annotate
   - Pick strategically: 10 with hands, 10 off-center, 10 unusual angles, 20 clear shots

3. **Annotate with AI assist**
   - Click the magic wand / auto-polygon tool
   - Roboflow's AI suggests a polygon; correct it with a few clicks
   - Accept when accurate, adjust when it includes hands/background
   - **Time:** ~10 seconds per mushroom

4. **Export**
   - Export → Format: **COCO**
   - Download the `.zip`

5. **Place in project**
   ```bash
   unzip ~/Downloads/roboflow-export.zip -d "data/Mushroom segmentation.coco-segmentation/"
   ```

6. **Convert to YOLO format**
   ```bash
   cd /path/to/project
   source .venv/bin/activate
   
   python scripts/convert_coco_to_yolo.py \
     --coco-json "data/Mushroom segmentation.coco-segmentation/train/_annotations.coco.json" \
     --images-dir "data/Mushroom segmentation.coco-segmentation/train" \
     --output-dir data/segmentation/eval_annotations/yolo \
     --rdp-epsilon 2.0
   ```

### How to Use the Annotations

**Option A — Evaluation ground truth (recommended for thesis):**
- Keep annotations separate from training
- Use them to measure YOLO vs. human quality
- Gives you objective IoU numbers to report

**Option B — Add to training:**
After running `prepare_yolo_seg_dataset.py`, copy the annotated images + labels into the train split:
```bash
cp "data/Mushroom segmentation.coco-segmentation/train/"*.jpg data/segmentation/images/train/
cp data/segmentation/eval_annotations/yolo/*.txt data/segmentation/labels/train/
```

**Best of both worlds:**
- Annotate ~50 images
- Put ~40 into training (higher quality labels)
- Hold out ~10 for evaluation (honest metrics)

---

### Optional: Include 12 Manual Annotations
The 12 COCO-annotated images are higher quality than SAM 2 pseudo-labels. You can use them as **evaluation ground truth** (default) or add them to training.

**To keep for evaluation (recommended for thesis metrics):**
Do nothing — they stay in `data/Mushroom segmentation.coco-segmentation/`.

**To add to YOLO training (better masks):**
After converting them to YOLO format (see Step 2b), manually copy the 12 image + label pairs into `data/segmentation/images/train/` and `data/segmentation/labels/train/`.

---

## Step 3 — YOLOv8 Segmentation (Google Colab)

### What It Does
Fine-tunes `yolov8n-seg.pt` (COCO-pretrained) to segment mushrooms using your SAM 2 masks as pseudo-labels, plus 48 background images to reduce false positives.

### Step 3A: Prepare Dataset Locally
```bash
cd /path/to/project
source .venv/bin/activate

# 1. Prepare YOLO dataset from SAM 2 masks + backgrounds
python scripts/prepare_yolo_seg_dataset.py \
  --images-dir data/raw/images \
  --masks-dir data/SegMaskSAM2 \
  --background-dir data/raw/background \
  --output-dir data/segmentation \
  --train-ratio 0.80 \
  --seed 42 \
  --bg-train 30 \
  --bg-val 10

# 2. (Optional) Add 12 manual annotations to training set
# Copy converted COCO annotations into train split:
cp "data/Mushroom segmentation.coco-segmentation/train/"*.jpg data/segmentation/images/train/ 2>/dev/null || true
cp data/segmentation/eval_annotations/yolo/*.txt data/segmentation/labels/train/ 2>/dev/null || true

# 3. Zip for Colab
rm -f data/segmentation.zip
zip -r data/segmentation.zip data/segmentation/
```

**Dataset composition after preparation:**
- `images/train/`: ~282 mushroom images + 30 background images
- `images/val/`: ~71 mushroom images + 10 background images
- `labels/train/` / `labels/val/`: YOLO polygon `.txt` files
- `dataset.yaml`: YOLO configuration

### Step 3B: Train in Google Colab

1. **Upload dataset**
   - Upload `data/segmentation.zip` to your Google Drive

2. **Open notebook**
   - Go to https://colab.research.google.com
   - Upload `scripts/colab_train_yolov8_seg.ipynb`

3. **Enable GPU**
   - Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save

4. **Run cells**
   - Cell 1: Install `ultralytics`
   - Cell 2: Mount Drive, set `DATASET_YAML` path
   - Cell 3: Verify dataset (should show ~312 train, ~80 val images)
   - Cell 4: Train (100 epochs, batch 8, imgsz 640, lr0 0.001)
   - Cell 5: Copy `best.pt` and metadata
   - Cell 6: Download `yolov8_seg_ft.pt`

5. **Place checkpoint in project**
   ```bash
   mv ~/Downloads/yolov8_seg_ft.pt /path/to/project/artifacts/
   ```

### Training Hyperparameters (for your thesis)
```python
model = YOLO('yolov8n-seg.pt')
results = model.train(
    data=DATASET_YAML,
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,
    close_mosaic=10,
    lr0=0.001,
    device=0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)
```

**Expected time:** 15–30 minutes on T4 GPU. Early stopping often kicks in around epoch 60.

### What to Document for Your Thesis
- Base model: `yolov8n-seg` (6.8 MB)
- Training data: 352 SAM 2 pseudo-labels + 48 backgrounds (+ optionally 12 manual annotations)
- Augmentation: HSV jitter, mosaic (disabled last 10 epochs)
- Final metrics from `runs/segment/train*/results.png`: `seg/mAP50`, `seg/mAP50-95`

---

## Step 4 — Evaluation & Promotion

### Compare Fine-Tuned vs Generic YOLO
```bash
cd /path/to/project
source .venv/bin/activate

python scripts/evaluate_segmentation.py \
  --model artifacts/yolov8_seg_ft.pt \
  --images-dir data/raw/evaluation_images \
  --masks-dir data/SegMaskSAM2_eval_holdout \
  --compare-generic \
  --output artifacts/segmentation_evaluation.json
```

This produces:
- Mean IoU, Precision, Recall
- Comparison against generic `yolov8n-seg.pt`
- Per-image breakdown

### Promotion Gate (switch to new model)
The fine-tuned model is promoted if:
1. It improves on ≥2 of 3 metrics (IoU, Precision, Recall)
2. `pytest tests/` passes with 0 failures
3. CPU inference is within 150% of generic model speed

### Verify CNN in Pipeline
```bash
pytest tests/test_cnn_classifier.py -v   # If such tests exist
pytest tests/ -v                           # Full regression suite
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| SAM 2 import error (`No module named sam2`) | Run from `/tmp`, not from project root (repo shadows package) |
| SAM 2 CUDA error on CPU | Set `device="cpu"` in script args |
| YOLO training says "no labels found" | Check `dataset.yaml` path; use absolute path in Colab |
| CNN warns "Missing folder for Other Boletus" | Ensure folder is named `BO.BA` (not `Brunsopp`) |
| Colab disconnects during training | Use Google Drive mount, not direct upload; training resumes from last epoch if you reconnect |
| Background images not included | Verify `--background-dir` path in `prepare_yolo_seg_dataset.py` |
| `prepare_yolo_seg_dataset.py` skips images | Ensure mask filename matches image stem (`<name>_sam2.png`) |

---

## Quick Reference Commands

```bash
# Full local pipeline (SAM 2 + dataset prep)
cd /tmp
python /path/to/project/scripts/generate_sam2_masks.py --project-root /path/to/project --images-dir data/raw/images --output-dir data/SegMaskSAM2

cd /path/to/project
source .venv/bin/activate
python scripts/prepare_yolo_seg_dataset.py --images-dir data/raw/images --masks-dir data/SegMaskSAM2 --background-dir data/raw/background --output-dir data/segmentation --train-ratio 0.80 --seed 42 --bg-train 30 --bg-val 10
zip -r data/segmentation.zip data/segmentation/

# CNN training
python scripts/train_cnn.py --epochs 20 --batch-size 8 --lr 3e-4

# Evaluation
python scripts/evaluate_segmentation.py --model artifacts/yolov8_seg_ft.pt --images-dir data/raw/evaluation_images --masks-dir data/SegMaskSAM2_eval_holdout --compare-generic --output artifacts/segmentation_evaluation.json
```

---

## File Checklist Before You Start

- [ ] `data/raw/images/` has 13 species folders (including `BO.BA`)
- [ ] `data/raw/background/` has 48 images
- [ ] `artifacts/sam2.1_hiera_tiny.pt` exists (156 MB)
- [ ] `.venv` is activated and has `torch`, `timm`, `ultralytics`, `sam2`
- [ ] Google Colab account ready
- [ ] Google Drive has space for ~200 MB dataset zip
