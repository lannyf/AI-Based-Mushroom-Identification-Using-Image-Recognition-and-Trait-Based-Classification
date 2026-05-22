# YOLOv8 Segmentation Retraining on Google Colab

## What you need

- Your Roboflow export: `Mushrooms.yolov8_v2.zip` (already in `data/Yolov8/`)
- A Google account (for Drive + Colab)
- ~1–2 hours of GPU time

---

## Step 1: Upload dataset to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder called `mushroom_yolo`
3. Upload `Mushrooms.yolov8_v2.zip` into that folder

---

## Step 2: Open Colab and run this notebook

Go to [Google Colab](https://colab.research.google.com) → **New Notebook**

Copy-paste the cells below one by one and run them.

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Unzip dataset
```python
import zipfile
import os

zip_path = '/content/drive/MyDrive/mushroom_yolo/Mushrooms.yolov8_v2.zip'
extract_path = '/content/mushroom_dataset'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted files:")
for f in os.listdir(extract_path):
    print(f"  {f}")
```

### Cell 3: Install Ultralytics
```python
!pip install -q ultralytics
```

### Cell 4: Verify GPU
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

If this prints `CUDA available: False`, go to **Runtime → Change runtime type → Hardware accelerator → GPU → Save**, then restart the runtime.

### Cell 5: Inspect data.yaml
```python
yaml_path = '/content/mushroom_dataset/data.yaml'
with open(yaml_path) as f:
    print(f.read())
```

### Cell 6: Fix paths in data.yaml (Colab needs absolute paths)
```python
yaml_path = '/content/mushroom_dataset/data.yaml'

with open(yaml_path, 'r') as f:
    lines = f.readlines()

with open(yaml_path, 'w') as f:
    for line in lines:
        if line.startswith('train:'):
            f.write(f'train: /content/mushroom_dataset/train/images\n')
        elif line.startswith('val:'):
            f.write(f'val: /content/mushroom_dataset/valid/images\n')
        elif line.startswith('test:'):
            f.write(f'test: /content/mushroom_dataset/test/images\n')
        else:
            f.write(line)

print("Updated data.yaml:")
with open(yaml_path) as f:
    print(f.read())
```

### Cell 7: Train YOLOv8n-seg
```python
from ultralytics import YOLO

# Load pretrained segmentation model
model = YOLO('yolov8n-seg.pt')

# Train
results = model.train(
    data='/content/mushroom_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='mushroom_seg_v2',
    project='/content/mushroom_dataset/runs',
    patience=20,        # early stopping if no improvement for 20 epochs
    save=True,
    device=0
)
```

Training will take ~30–90 minutes depending on dataset size and GPU.

### Cell 8: Validate
```python
metrics = model.val()
print(f"mAP50: {metrics.seg.map50:.3f}")
print(f"mAP50-95: {metrics.seg.map:.3f}")
```

### Cell 9: Copy best.pt back to Drive
```python
import shutil

src = '/content/mushroom_dataset/runs/mushroom_seg_v2/weights/best.pt'
dst = '/content/drive/MyDrive/mushroom_yolo/best_v2.pt'

shutil.copy(src, dst)
print(f"Saved best model to: {dst}")
```

### Cell 10: (Optional) Copy last.pt as backup
```python
shutil.copy(
    '/content/mushroom_dataset/runs/mushroom_seg_v2/weights/last.pt',
    '/content/drive/MyDrive/mushroom_yolo/last_v2.pt'
)
print("Backup saved.")
```

---

## Step 3: Download model back to your machine

After training completes:
1. Go to Google Drive → `mushroom_yolo/`
2. Download `best_v2.pt`
3. Place it in your project:
   ```
   data/Yolov8/best_v2.pt
   ```

---

## Step 4: Update your pipeline

Edit `models/mushroom_segmenter.py` (or wherever you load the model) to point to the new weights:

```python
# Old
preferred_path="data/Yolov8/best.pt"

# New
preferred_path="data/Yolov8/best_v2.pt"
```

---

## Expected training metrics

With your cleaned single-mushroom dataset (~200+ images), you should see:
- **mAP50**: 0.60–0.80 (depending on class balance)
- **mAP50-95**: 0.35–0.55

If mAP50 is below 0.50 after 100 epochs, you likely need more training data or the annotations have quality issues.

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size in Cell 7:
```python
batch=8   # or even 4
```

### "Dataset not found"
Make sure `data.yaml` paths were updated in Cell 6. Roboflow exports use relative paths by default which break in Colab.

### Training is very slow
- Confirm GPU is enabled (Cell 4 should show a GPU name like "T4" or "L4")
- If on CPU, training will take 10x longer

### Want to resume training?
```python
model = YOLO('/content/mushroom_dataset/runs/mushroom_seg_v2/weights/last.pt')
model.train(data='/content/mushroom_dataset/data.yaml', epochs=50, resume=True)
```
