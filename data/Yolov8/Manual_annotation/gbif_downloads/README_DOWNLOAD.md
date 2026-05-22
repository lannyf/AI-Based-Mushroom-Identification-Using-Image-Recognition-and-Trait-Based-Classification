# GBIF Training Image Download — Summary

## What was downloaded

**154 mushroom images** from GBIF (Global Biodiversity Information Facility) to address critical gaps in the YOLOv8 segmentation training dataset.

## Where they are

All images are in this folder:
```
data/Yolov8/Manual_annotation/gbif_downloads/
```

A manifest (`gbif_manifest.csv`) records attribution, license, and source URL for each image.

## Species distribution

| Species ID | Scientific name | Count | Purpose |
|-----------|----------------|-------|---------|
| **PL.OS** | *Pleurotus ostreatus* | 12 | **Missing entirely** from training. White gilled bracket fungus on wood. |
| **CR.CO** | *Craterellus cornucopioides* | 12 | **Missing entirely** from training. Black trumpet (Coral class candidate or new class). |
| **LY.PE** | *Lepista personata* | 12 | **Missing entirely** from training. Blewit, pinkish gilled, mislabeled as Coral in benchmark. |
| **CO.CO** | *Coprinus comatus* | 10 | Severely underrepresented (only 2 in current training). Shaggy ink cap. |
| **CAL.GI** | *Calocybe gambosa* | 10 | Severely underrepresented. St George's mushroom, white gilled. |
| **FO.BE** | *Fomitopsis betulina* | 10 | Severely underrepresented. Birch polypore, bracket fungus. |
| **GY.ES** | *Gyromitra esculenta* | 10 | Severely underrepresented. False morel, brain-like cap. |
| **GA.MA** | *Ganoderma applanatum* | 10 | Severely underrepresented. Artist's bracket, shelf fungus. |
| **HY.RE** | *Hydnum repandum* | 8 | Weak representation. Hedgehog mushroom, spines instead of gills. |
| **CA.TU** | *Cantharellus tubaeformis* | 8 | Weak representation. Trumpet chanterelle, yellow-brown. |
| **AM.MU** | *Amanita muscaria* | 10 | Extra below-view / alternate angles. Was already in training. |
| **AM.VI** | *Amanita virosa* | 8 | Extra below-view / alternate angles. Was already in training. |
| **BO.BA** | *Imleria badia* | 8 | Extra below-view / alternate angles. Was already in training. |
| **BO.ED** | *Boletus edulis* | 8 | Extra below-view / alternate angles. Was already in training. |
| **CA.CI** | *Cantharellus cibarius* | 6 | Weak detection confidence. Golden chanterelle. |
| **LA.HE** | *Lactarius helvus* | 6 | Mislabeled as Coral in benchmark. Orange milkcap. |
| **RU.BA** | *Russula badia* | 6 | Over-detection issues. Purple/brown brittlegill. |

## What problems this addresses

### 1. Complete misses (9 benchmark images → 0 detections)
**Root cause:** Species had 0-1 training images.  
**Fix:** Added 10-12 images each for PL.OS, CR.CO, LY.PE, CO.CO, CAL.GI, FO.BE, GY.ES, GA.MA.

### 2. Below-view mislabeled as Cap (13 images, 10 species)
**Root cause:** Model never learned to distinguish below-view gill/pore shots from top-down cap shots.  
**Fix:** Added extra images for AM.MU, AM.VI, BO.BA, BO.ED — prioritize **below-view / underside-visible** photos when annotating. In Roboflow, label these as `Underside` (or `Cap` + `Underside` combo where visible).

### 3. Classical mushrooms mislabeled as Coral (11 images, 8 species)
**Root cause:** Coral class overrepresented (50 instances) vs classical gilled mushrooms (30).  
**Fix:** Added classical gilled mushrooms (LY.PE, CAL.GI, CO.CO, LA.HE) and bracket fungi (FO.BE, GA.MA) to re-balance.

### 4. Over-detection / spurious boxes (10 images)
**Root cause:** Training included cluster photos with 5-20 mushrooms in one frame.  
**Fix:** New images are mostly single-mushroom photos. When annotating, **prefer single-mushroom shots** and avoid annotating every tiny fragment.

## Annotation instructions for Roboflow

1. **Upload** the contents of `gbif_downloads/` (or the whole `Manual_annotation/` folder) to Roboflow.
2. **Label using 4 classes:** `Cap`, `Stem`, `Underside`, `Coral`
3. **Critical — below-view policy:**
   - If the photo is taken from **below/angle showing gills or pores**, label the main mushroom part as **`Underside`** (even if some cap rim is visible).
   - If the stem is clearly visible in the below shot, add a separate **`Stem`** label.
   - This directly fixes the "below→Cap" mislabeling bug.
4. **Bracket / polypore policy:**
   - Shelf/bracket fungi growing on wood (FO.BE, GA.MA): label as **`Coral`** if they have that branching/coral-like morphology, or **`Cap`** if they have a flat cap surface with pores underneath.
   - For these species, also try to get some **pore-surface (underside)** shots labeled as `Underside`.
5. **Single-mushroom priority:**
   - If an image has multiple mushrooms, annotate **only the clearest 1-2 specimens**.
   - Skip photos with 5+ overlapping mushrooms entirely (or mark them for exclusion).
6. **Export format:**
   - YOLOv8 segmentation
   - Train/valid/test split: **80/10/10** (Roboflow auto-split is fine)
   - Include existing annotated images + these new ones

## Next steps after annotation

1. Export from Roboflow → replace `mushroom segmentation.yolov8(1)/` folder
2. Retrain YOLOv8:
   ```bash
   .venv/bin/python3 -m ultralytics.models.yolo.segment.train \
     data=data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
   ```
3. Rerun benchmark to verify fixes.

## Important: Image quality

All downloaded images were verified to be actual photographs (not diagrams/maps), with dimensions between 768×1024 and 2048×2048 pixels. No unreadable or corrupted files. File sizes range from ~100KB to ~4MB.

## Attribution

All images sourced from GBIF occurrence records. License information for each image is recorded in `gbif_manifest.csv`. Common licenses include CC-BY, CC-BY-NC, CC0. Verify individual licenses before public redistribution.
