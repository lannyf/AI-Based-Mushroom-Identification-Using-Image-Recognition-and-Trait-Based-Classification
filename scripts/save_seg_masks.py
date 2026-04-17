"""
Save cleaned segmentation masks and metadata for images under data/raw/images.
Saves PNG masks to data/SegMask and writes a JSON manifest to data/SegMaskJS/seg_masks.json.

Run with the project venv: 
/home/iannyf/mushroom-venv/bin/python scripts/save_seg_masks.py
"""
from pathlib import Path
import hashlib
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mushroom_segmenter import get_segmenter

ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = ROOT / "data" / "raw" / "images"
OUT_MASK_DIR = ROOT / "data" / "SegMask"
OUT_JSON_DIR = ROOT / "data" / "SegMaskJS"
OUT_JSON_PATH = OUT_JSON_DIR / "seg_masks.json"

OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Use small model by default for speed; change to artifacts checkpoint if available
seg = get_segmenter("yolov8n-seg")

records = []

image_paths = list(IMAGE_ROOT.rglob("*.jpg"))
print(f"Found {len(image_paths)} images to process")
start = time.time()
for p in image_paths:
    rel = p.relative_to(ROOT)
    try:
        b = p.read_bytes()
    except Exception as e:
        print("Failed to read", p, e)
        continue
    sha1 = hashlib.sha1(b).hexdigest()
    mask_fname = f"{sha1}.png"
    mask_path = OUT_MASK_DIR / mask_fname
    # Skip if exists
    if mask_path.exists():
        print("Skipping (exists)", p)
        # still load minimal record
        records.append({"image": str(rel), "sha1": sha1, "mask_path": str(mask_path)})
        continue
    try:
        res = seg.segment(b, top_n=3)
        insts = res.get("instances", [])
        sel_idx = res.get("selected_index")
        if sel_idx is None or not insts:
            print("No instances for", p)
            records.append({"image": str(rel), "sha1": sha1, "mask_path": None})
            continue
        sel = insts[sel_idx]
        mask = sel.get("cleaned_mask")
        if mask is None:
            records.append({"image": str(rel), "sha1": sha1, "mask_path": None})
            continue
        # write mask
        import cv2
        cv2.imwrite(str(mask_path), mask)
        rec = {
            "image": str(rel),
            "sha1": sha1,
            "mask_path": str(mask_path.relative_to(ROOT)),
            "bbox": sel.get("bbox"),
            "model_confidence": float(sel.get("model_confidence", 0.0)),
            "area_ratio": float(sel.get("area_ratio", 0.0)),
            "fragment_count": int(sel.get("fragment_count", 0)),
            "hole_ratio": float(sel.get("hole_ratio", 0.0)),
            "boundary_irregularity": float(sel.get("boundary_irregularity", 0.0)),
        }
        records.append(rec)
        print("Saved", mask_path, "for", p)
    except Exception as exc:
        print("Segmentation error for", p, exc)

# write manifest
with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"Processed {len(image_paths)} images in {time.time()-start:.1f}s. Manifest at {OUT_JSON_PATH}")
