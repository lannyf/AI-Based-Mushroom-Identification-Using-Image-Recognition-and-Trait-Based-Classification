# YOLOv8 Segmentation Training Guide

Train a YOLOv8n-seg model to segment mushroom parts (cap, underside, stem, coral) from photos.

---

## Overview

| Setting | Value |
|---------|-------|
| Base model | `yolov8n-seg.pt` (nano, fastest) |
| Task | Instance segmentation (4 classes) |
| Input size | 640×640 |
| Training data | Roboflow manual annotations |
| Best checkpoint | `data/Yolov8/best.pt` |

---

## Dataset Structure

The segmentation dataset is exported from **Roboflow** in YOLOv8 format:

```
data/Yolov8/
├── best.pt                    # Current best model (copy of trained checkpoint)
├── dataset.yaml               # Dataset configuration
├── train/
│   ├── images/                # Training images
│   └── labels/                # YOLO segmentation masks (.txt)
├── valid/
│   ├── images/                # Validation images
│   └── labels/
└── test/
    ├── images/                # Test images
    └── labels/
```

**Classes:**
```yaml
names:
  0: mushroom        (whole mushroom)
  1: cap             (cap/top view)
  2: underside       (gills/pores/folds)
  3: stem            (stipe/stalk)
  4: coral           (coral-like structures)
```

---

## Option A: Train Locally (CPU — Very Slow)

```bash
cd ~/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification

python scripts/train_yolov8_seg.py \
  --dataset data/Yolov8/dataset.yaml \
  --device cpu \
  --epochs 100 \
  --batch 8
```

**Expected time:** 10–24 hours on CPU (not recommended).

---

## Option B: Train on Google Colab (Recommended)

### Step 1: Prepare the Dataset

Zip your YOLOv8 dataset folder:

```bash
cd ~/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification

zip -r ~/yolov8-dataset.zip data/Yolov8/
```

Upload `yolov8-dataset.zip` to your **Google Drive root**.

### Step 2: Open Colab Notebook

Use the existing notebook: `scripts/colab_train_yolov8_seg.ipynb`

Or create a new notebook with these cells:

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: Extract dataset
!unzip -q /content/drive/MyDrive/yolov8-dataset.zip -d /content/
```

```python
# Cell 3: Install ultralytics
!pip install -q ultralytics
import ultralytics
print('Ultralytics:', ultralytics.__version__)
```

```python
# Cell 4: Train
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

results = model.train(
    data='/content/data/Yolov8/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device=0,  # GPU
    project='artifacts/yolov8_seg_runs',
    name='mushroom_seg',
)
```

```python
# Cell 5: Copy best checkpoint
import shutil
from pathlib import Path

src = Path('artifacts/yolov8_seg_runs/mushroom_seg/weights/best.pt')
dst = Path('/content/drive/MyDrive/yolov8_best.pt')

if src.exists():
    shutil.copy2(src, dst)
    print(f"Saved to: {dst}")
else:
    print("best.pt not found")
```

### Step 3: Configure Runtime

**Runtime → Change runtime type → T4 GPU**

### Step 4: Run and Download

Training takes **1–2 hours** on T4 GPU for 100 epochs with ~640 images.

After training:
1. Download `yolov8_best.pt` from Drive
2. Copy it to your project: `cp ~/Downloads/yolov8_best.pt data/Yolov8/best.pt`

---

## Option C: Train with the Project Script (GPU or CPU)

```bash
cd ~/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification

# On GPU
python scripts/train_yolov8_seg.py \
  --dataset data/Yolov8/dataset.yaml \
  --device 0 \
  --epochs 100 \
  --batch 16 \
  --dest data/Yolov8/best.pt

# On CPU
python scripts/train_yolov8_seg.py \
  --dataset data/Yolov8/dataset.yaml \
  --device cpu \
  --epochs 100 \
  --batch 8 \
  --dest data/Yolov8/best.pt
```

### Script Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | Path to `dataset.yaml` |
| `--device` | `cpu` | `cpu`, `0` (first GPU), `0,1,2,3` (multi-GPU) |
| `--epochs` | 100 | Training epochs |
| `--imgsz` | 640 | Input image size |
| `--batch` | 8 | Batch size (use 16 on Colab T4) |
| `--patience` | 20 | Early stopping patience |
| `--lr0` | 0.001 | Initial learning rate |
| `--dest` | `artifacts/yolov8_seg_ft.pt` | Where to save best checkpoint |

---

## Training Hyperparameters

The script uses these defaults (tuned for the mushroom dataset):

```python
epochs=100
imgsz=640
batch=8          # increase to 16 on GPU
patience=20      # stop if no val improvement for 20 epochs
close_mosaic=10  # disable mosaic augmentation for last 10 epochs
lr0=0.001        # conservative LR for fine-tuning
hsv_h=0.015      # HSV hue augmentation
hsv_s=0.7        # HSV saturation augmentation
hsv_v=0.4        # HSV value augmentation
```

---

## After Training

The script automatically:
1. Saves the best checkpoint to `--dest`
2. Writes metadata JSON next to the checkpoint

```
artifacts/
├── yolov8_seg_ft.pt              # Best model weights
├── yolov8_seg_ft_metadata.json   # Training config & dataset info
└── yolov8_seg_runs/              # Full training run logs
    └── mushroom_seg/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.png
        └── confusion_matrix.png
```

Copy the best checkpoint to where the pipeline expects it:

```bash
cp artifacts/yolov8_seg_ft.pt data/Yolov8/best.pt
```

---

## Expected Results

With the current Roboflow dataset (~640 images, 4 classes):

| Metric | Target | Notes |
|--------|--------|-------|
| mAP@50 | 0.75–0.85 | Overall segmentation quality |
| Cap mAP | 0.80–0.90 | Usually the strongest class |
| Underside mAP | 0.60–0.75 | Weakest class; visual confusion with cap |
| Stem mAP | 0.70–0.80 | Moderate performance |

The **Underside** class is consistently the weakest due to visual similarity with Cap and fewer training examples.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `dataset.yaml not found` | Check path relative to project root |
| `CUDA out of memory` | Reduce `--batch` to 8 or 4 |
| Training very slow | Verify GPU with `nvidia-smi`; use `--device 0` |
| Low mAP on underside | Add more underside-only images to training set |
| `ultralytics` not installed | `pip install ultralytics` |
| Roboflow export is wrong format | Re-export in **YOLOv8** format (not YOLOv5 or COCO) |

---

## Re-training from Scratch

If you want to completely restart training (not resume):

```bash
# Delete old runs
rm -rf artifacts/yolov8_seg_runs/

# Train fresh
python scripts/train_yolov8_seg.py \
  --dataset data/Yolov8/dataset.yaml \
  --device 0 \
  --epochs 100
```
