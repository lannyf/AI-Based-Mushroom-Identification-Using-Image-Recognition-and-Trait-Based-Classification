# Manual Annotation Set for YOLOv8 Segmentation

This folder contains **50 images*** selected for manual polygon annotation in Roboflow.
These annotations will be used to improve the YOLOv8 segmentation model beyond the current SAM2-generated pseudo-labels.

## Selection Rationale

Images were chosen algorithmically based on SAM2 mask quality metrics, prioritizing cases where the automatic pipeline struggled:

- **22 images** used fallback strategies (`bbox_fallback` or `amg_retry`) — SAM2's primary center-point prompt failed, indicating unusual composition (off-center, partially cropped, hands/occlusion, multiple objects).
- **28 images** had low prompt overlap, extreme area ratios, or poor compactness scores — indicating masks that likely miss parts of the mushroom or include background.

This prioritizes **visual difficulty over species balance**, which is appropriate because YOLOv8 is trained as a single-class "mushroom" detector.

## Species Distribution

| Species Code | Count | Notes |
|--------------|-------|-------|
| AM.MU (Fly Agaric) | 7 | |
| AM.VI (Amanita virosa) | 6 | |
| HY.PS (False Chanterelle) | 6 | |
| CR.CO (Black Trumpet) | 6 | |
| coprinus_comatus | 0 | |
| BO.ED (Porcini) | 6 | |
| BO.BA (Other Boletus) | 5 | |
| CA.CI (Chanterelle) | 3 | |
| fomitopsis_betulina | 0 | |
| lycoperdon_utriforme | 0 | |
| ramaria_botrytis | 0 | |
| sparassis_crispa | 0 | |

## Annotation Rules (Roboflow)

1. **Class**: Single class — `mushroom`
2. **Include**: Cap, stem, and visible gills/pores
3. **Exclude**: Hands, fingers, soil, grass, leaves, other objects
4. **Edge handling**: Trace the outer contour precisely; include fine structures if clearly part of the fruiting body
5. **Multiple mushrooms**: If multiple clearly separated fruiting bodies are present, annotate each with its own polygon
6. **Partial occlusion**: If a hand covers part of the mushroom, annotate only the visible portion (do not guess hidden areas)

## Roboflow Workflow

1. Go to [roboflow.com](https://roboflow.com) → Create new project → Segmentation
2. Upload all images from this folder
3. Use **AI-assisted polygon tool** (Smart Polygon) — expect ~10–30 sec per image
4. Review each mask carefully against the rules above
5. When done, export as **COCO Segmentation** format
6. Convert to YOLO polygon format:

```bash
python scripts/convert_coco_to_yolo.py \
    --coco-json "data/Mushroom segmentation.coco-segmentation/train/_annotations.coco.json" \
    --images-dir "data/Mushroom segmentation.coco-segmentation/train" \
    --output-dir data/segmentation/manual_annotations/yolo \
    --rdp-epsilon 2.0
```

## File Inventory

See `selected_for_annotation.json` for the full list with SAM2 quality metadata (priority score, strategy, prompt overlap, area ratio, compactness, SAM score). This metadata explains why each image was selected — higher scores mean the automatic mask was more uncertain.

## Next Steps After Annotation

1. Convert COCO → YOLO (command above)
2. Merge these manual annotations with the existing SAM2 pseudo-labels:
   - Option A: Use manual annotations as training data alongside SAM2 labels
   - Option B: Replace SAM2 labels for these 50 images with manual ones
3. Retrain YOLOv8n-seg in Colab with the combined dataset
4. Evaluate on the separate holdout evaluation set

---

*Selected automatically on 2026-05-08 using SAM2 mask quality heuristics.*
