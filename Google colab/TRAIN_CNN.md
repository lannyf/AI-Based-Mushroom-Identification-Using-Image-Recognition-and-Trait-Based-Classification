# CNN Classifier Training Guide

Fine-tune an EfficientNet-B3 classifier for mushroom species recognition from single images.

---

## Overview

| Setting | Value |
|---------|-------|
| Base model | `efficientnet_b3` (timm) |
| Pretrained weights | ImageNet |
| Input size | 300×300 |
| Classes | 16 mushroom species |
| Best weights | `artifacts/cnn_weights.pt` |

---

## Dataset Structure

Training images are organized by species folder in `data/raw/images/`:

```
data/raw/images/
├── AM.MU/          # Fly Agaric (Amanita muscaria)
│   ├── img001.jpg
│   └── ...
├── CA.CI/          # Chanterelle (Cantharellus cibarius)
│   └── ...
├── BO.ED/          # Porcini (Boletus edulis)
│   └── ...
├── CR.CO/          # Black Trumpet (Craterellus cornucopioides)
│   └── ...
└── ... (16 species total)
```

**Minimum:** 10 images per species  
**Recommended:** 20–40 images per species for solid fine-tuning  
**Current dataset:** ~640 images across 16 species

---

## Configuration

All training settings live in `config/image_model_config.py`:

```python
BASE_MODEL = "efficientnet_b3"
INPUT_SIZE = (300, 300)
RESIZE_SIZE = 320
NUM_CLASSES = 16

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 3e-4
VAL_FRACTION = 0.2          # 20% validation split
EARLY_STOPPING_PATIENCE = 5

# Phase 1: train head only (1/3 of epochs)
HEAD_ONLY_EPOCHS_FRACTION = 1/3
HEAD_ONLY_LR_FACTOR = 1.0

# Phase 2: fine-tune full network
FINETUNE_LR_FACTOR = 0.1    # 10× lower LR for full fine-tuning
```

---

## Option A: Train Locally

### With GPU (recommended if you have CUDA)

```bash
cd ~/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification

python scripts/train_cnn.py
```

### With CPU (slow but functional)

```bash
python scripts/train_cnn.py --epochs 20 --batch-size 4
```

**Expected time:**
- GPU (RTX 3060): ~10–15 minutes
- CPU (Ryzen 7): ~1–2 hours

---

## Option B: Train on Google Colab

### Step 1: Prepare Images

Zip the training images:

```bash
cd ~/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification

zip -r ~/cnn-images.zip data/raw/images/
```

Upload `cnn-images.zip` to your **Google Drive root**.

### Step 2: Colab Notebook

Create a new notebook with these cells:

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: Extract images
import shutil
from pathlib import Path

src = Path('/content/drive/MyDrive/cnn-images.zip')
dst = Path('/content/mushroom-project')

if src.exists():
    shutil.unpack_archive(str(src), str(dst))
    print(f"Extracted to {dst}")
else:
    raise FileNotFoundError(f"Upload {src.name} to Drive root first")
```

```python
# Cell 3: Install dependencies
!pip install -q timm torch torchvision pillow
```

```python
# Cell 4: Clone/pull project code
# Option A: If you have the project in Drive
!cp -r /content/drive/MyDrive/mushroom-project/* /content/mushroom-project/

# Option B: Clone from GitHub
# !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /content/mushroom-project
```

```python
# Cell 5: Train
%cd /content/mushroom-project

import sys
sys.path.insert(0, '/content/mushroom-project')

!python scripts/train_cnn.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-4
```

```python
# Cell 6: Copy weights back to Drive
import shutil
from pathlib import Path

weights = Path('/content/mushroom-project/artifacts/cnn_weights.pt')
drive_dst = Path('/content/drive/MyDrive/cnn_weights.pt')

if weights.exists():
    shutil.copy2(weights, drive_dst)
    print(f"Saved to: {drive_dst}")
else:
    print("Weights not found — training may have failed")
```

### Step 3: Configure Runtime

**Runtime → Change runtime type → T4 GPU**

### Step 4: Run

Training takes **5–10 minutes** on T4 GPU for 30 epochs.

After training:
1. Download `cnn_weights.pt` from Drive
2. Copy to your project: `cp ~/Downloads/cnn_weights.pt artifacts/cnn_weights.pt`

---

## Training Script Arguments

```bash
python scripts/train_cnn.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 20 | Total training epochs |
| `--batch-size` | 8 | Batch size (use 16 on GPU) |
| `--lr` | 3e-4 | Initial learning rate |

---

## Two-Phase Training

The script uses a standard transfer learning approach:

### Phase 1: Train Head Only (first 1/3 of epochs)

```python
# Freeze backbone
for name, param in model.named_parameters():
    param.requires_grad = "classifier" in name or "head" in name

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
```

Only the classification head learns. The pretrained ImageNet backbone stays frozen.

### Phase 2: Fine-Tune Full Network (remaining 2/3 of epochs)

```python
# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

optimizer = Adam(model.parameters(), lr=3e-5)  # 10× lower LR
```

The entire network is fine-tuned at a reduced learning rate to avoid catastrophic forgetting.

---

## Data Augmentation

Training uses these augmentations (defined in `config/image_model_config.py`):

```python
TRAIN_AUGMENTATION = {
    "random_resized_crop_scale": (0.7, 1.0),  # Random zoom
    "color_jitter": (0.3, 0.3, 0.2),          # Brightness, contrast, saturation
    "random_rotation": 20,                     # ±20° rotation
    "random_horizontal_flip": True,            # 50% horizontal flip
}
```

Validation uses only resize + center crop + normalization.

---

## Normalization

All images are normalized with ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## After Training

The script saves:

```
artifacts/
├── cnn_weights.pt                # Best model (highest val accuracy)
└── cnn_training_history.json     # Loss/accuracy per epoch
```

The API and benchmark will automatically pick up `artifacts/cnn_weights.pt` on next restart.

### View Training History

```python
import json
import matplotlib.pyplot as plt

with open('artifacts/cnn_training_history.json') as f:
    history = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='train')
axes[0].plot(history['val_loss'], label='val')
axes[0].set_title('Loss')
axes[0].legend()

axes[1].plot(history['train_acc'], label='train')
axes[1].plot(history['val_acc'], label='val')
axes[1].set_title('Accuracy')
axes[1].legend()

plt.show()
```

---

## Expected Results

With ~640 images across 16 species:

| Metric | Target | Notes |
|--------|--------|-------|
| Train accuracy | 85–95% | May overfit with small dataset |
| Val accuracy | 65–80% | More realistic generalization |
| Best val accuracy | ~75% | Saved as `cnn_weights.pt` |

If validation accuracy is much lower than training accuracy, the model is overfitting. Solutions:
- Add more training images
- Increase data augmentation
- Reduce epochs
- Use stronger regularization

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No images found` | Check `data/raw/images/` has species subfolders |
| `CUDA out of memory` | Reduce `--batch-size` to 4 or 2 |
| `timm not installed` | `pip install timm` |
| Training accuracy stuck at ~6% | Check species folders are correctly named |
| Validation accuracy < 50% | Add more images; check for label errors |
| `ModuleNotFoundError: config` | Run from project root, not `scripts/` |

---

## Re-training from Scratch

To discard previous weights and train fresh:

```bash
# Delete old weights
rm artifacts/cnn_weights.pt
rm artifacts/cnn_training_history.json

# Train new model
python scripts/train_cnn.py --epochs 30 --batch-size 8
```

The script always starts from ImageNet pretrained weights (not from your previous `cnn_weights.pt`).
