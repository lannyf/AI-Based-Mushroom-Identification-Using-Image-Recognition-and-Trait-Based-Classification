# YOLOv8 Retraining on Colab (No Drive Mount)

## Step 1: Upload ZIP to Colab

1. Open [Google Colab](https://colab.research.google.com) → **New Notebook**
2. In the left sidebar, click the **Files** icon (📁)
3. Drag your ZIP file into the file browser
4. Wait for upload to finish

## Step 2: Run these cells

### Cell 1 — Unzip dataset (auto-finds any .zip)
```python
import zipfile
import os

# Find any .zip file in /content/
zip_files = [f for f in os.listdir('/content/') if f.endswith('.zip')]

if not zip_files:
    print("No ZIP file found. Upload it first using the Files sidebar.")
else:
    zip_path = os.path.join('/content/', zip_files[0])
    print(f"Found: {zip_files[0]}")
    zipfile.ZipFile(zip_path, 'r').extractall('/content/mushroom_dataset')
    print("Extracted to /content/mushroom_dataset/")
```

### Cell 2 — Fix data.yaml paths
```python
import os

yaml_path = '/content/mushroom_dataset/data.yaml'

if not os.path.exists(yaml_path):
    # Try searching one level deeper (some exports nest the files)
    for root, dirs, files in os.walk('/content/mushroom_dataset/'):
        if 'data.yaml' in files:
            yaml_path = os.path.join(root, 'data.yaml')
            break

with open(yaml_path, 'r') as f:
    lines = f.readlines()

base_dir = os.path.dirname(yaml_path)

with open(yaml_path, 'w') as f:
    for line in lines:
        if line.startswith('train:'):
            f.write(f'train: {base_dir}/train/images\n')
        elif line.startswith('val:'):
            f.write(f'val: {base_dir}/valid/images\n')
        elif line.startswith('test:'):
            f.write(f'test: {base_dir}/test/images\n')
        else:
            f.write(line)

print(f"Fixed: {yaml_path}")
with open(yaml_path) as f:
    print(f.read())
```

### Cell 3 — Install ultralytics & check GPU
```bash
!pip install -q ultralytics
```
```python
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU — go to Runtime → Change runtime type → GPU")
```

### Cell 4 — Train
```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(
    data='/content/mushroom_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='mushroom_seg',
    patience=20
)
```

Training takes ~30–90 min.

### Cell 5 — Auto-download best.pt and last.pt
```python
from google.colab import files
import os

weights_dir = '/content/runs/segment/mushroom_seg/weights'

for fname in ['best.pt', 'last.pt']:
    path = os.path.join(weights_dir, fname)
    if os.path.exists(path):
        print(f"Downloading {fname} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)...")
        files.download(path)
    else:
        print(f"WARNING: {fname} not found at {path}")

print("\nDone. Check your browser's downloads folder.")
```

## Step 3: Rename and move

After the downloads finish, rename:
- `best.pt` → `best_v2.pt`
- `last.pt` → `last_v2.pt` (optional backup)

Place them in your project under `data/Yolov8/`.

## Optional: resume training

If Colab disconnects, re-upload the ZIP, then:
```python
from ultralytics import YOLO
model = YOLO('runs/segment/mushroom_seg/weights/last.pt')  # if still in session
model.train(data='/content/mushroom_dataset/data.yaml', epochs=50, resume=True)
```

> **Note:** If the runtime disconnected and files were lost, you must re-upload the ZIP and start from `best.pt` or `last.pt` if you saved them locally.
