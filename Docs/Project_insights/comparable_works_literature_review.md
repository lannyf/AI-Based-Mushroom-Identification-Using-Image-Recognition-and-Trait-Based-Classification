# Literature Review: Comparable AI Mushroom Identification Works

**Date:** 2026-05-15
**Purpose:** Summarize comparable works from the initial project plan and related literature found through web search, with applicability assessment to the current pipeline's accuracy blockers.

---

## 1. Comparable Works Cited in Initial Project Plan

The initial project plan (`Projektplan reviderad.md`, `01-literature-review.md`) cites two mushroom-specific AI works in detail.

### 1.1 Lee et al. (2022) — Smartphone CNN Classification
**Citation:** Lee, J. J., Aime, M. C., Rajwa, B., & Bae, E. (2022). Machine Learning-Based Classification of Mushrooms Using a Smartphone Application. *Applied Sciences*, 12(22), 11685. Purdue University.

**What they did:**
- Built an Android app (PUMA) that classifies mushroom genus from 224x224 RGB field photos.
- Trained three case studies: 2-class (Gyromitra vs. Morchella), 3-class (Clavulina, Inocybe, Marasmius), and 5-class (Agaricus, Amanita, Cantharellus, Pleurotus, Tricholoma).
- **2-class/3-class:** Custom shallow CNN (6 conv layers + 3 FC layers, 70% dropout).
- **5-class:** Switched to **fine-tuned ResNet-152** because the shallow CNN failed.
- Model trained on server, transferred to phone. Users get probability scores per genus.
- **Results:** 89-100% sensitivity/specificity on field images with diverse backgrounds.

**Key characteristic:** Pure end-to-end image classifier. No segmentation, no part detection, no explicit trait extraction.

### 1.2 Chaschatzis et al. (2025) — UAV + YOLOv5 Wild Mushroom Detection
**Citation:** Chaschatzis, C., Karaiskou, C., Iakovidou, C., Radoglou-Grammatikis, P., Bibi, S., Goudos, S. K., & Sarigiannidis, P. G. (2025). Detection of Wild Mushrooms Using Machine Learning and Computer Vision. *Information*, 16(7), 539. University of Western Macedonia, Greece.

**What they did:**
- Built a UAV/drone system for locating wild mushrooms in forests, focusing on *Macrolepiota procera*.
- Uses **YOLOv5 object detection** with a custom dataset (WOES: 907 aerial + ground images).
- **Two-phase pipeline:**
  1. Phase 1: Drone flies at 40-100m with multispectral camera. Computes NDRE vegetation index, uses K-means + contour detection to find potential mushroom patches.
  2. Phase 2: Drone descends below 20m. Runs YOLOv5 on live RGB stream for verification.
- Two models used sequentially: first detects "wild mushroom," second identifies *Macrolepiota procera* by looking for dark central mottling.
- **Image preprocessing:** Extreme brightness/contrast adjustment (brightness 0.1, contrast 10.0) to highlight mottling before second model.
- **Hyperparameter tuning:** Genetic algorithm evolution (1000 iterations) on YOLOv5 training — improved mAP by 2-5%.
- **Results:** >90% accuracy, 30 min per field survey.

**Key characteristic:** Detection/localization system, not fine-grained species identifier. Bounding-box YOLO only. Distinguishes one visual feature (mottling) to separate two species.

---

## 2. Applicability Assessment to Current Pipeline Blockers

Current accuracy blockers:
1. Color vocabulary mismatch (10 coarse HSV bins vs. DB fine gradations)
2. 2D contour shape unreliability (viewpoint-dependent, defaults to "wavy")
3. Hymenophore taxonomy mismatch (surface type vs. attachment type)

| Method from Comparable Work | Applicability to Blockers | Assessment |
|-----------------------------|---------------------------|------------|
| Lee et al. end-to-end CNN | Color, Shape, Hymenophore | LOW -- Implicit feature learning; no named traits or vocabulary mapping. |
| Chaschatzis YOLOv5 detection | Color, Shape, Hymenophore | LOW -- Bounding-box only; no part-level trait extraction. |
| Chaschatzis multispectral NDRE | Color | NONE for smartphone -- Requires Parrot Sequoia+ camera. |
| Chaschatzis two-stage pipeline | General architecture | MODERATE -- Mirrors our YOLO -> masks -> trait extractor concept. |
| Chaschatzis species-specific preprocessing | Color, Hymenophore | MODERATE -- Preprocessing to highlight diagnostic features could be adapted for gill enhancement. |
| Chaschatzis GA hyperparameter evolution | YOLO mask quality | MODERATE -- Our YOLOv8 trained with defaults; evolution could improve mask IoU. |
| Lee et al. ResNet-152 fine-tuning | Image CNN module | MODERATE -- Validates switching to deeper models if MobileNetV2 struggles. |

**Conclusion:** The comparable works in the initial project plan are end-to-end classification/detection systems that bypass explicit trait extraction entirely. Neither addresses fine-grained morphological trait naming. Our pipeline is actually **more ambitious** than both cited works.

---

## 3. Extended Literature Search: Methods for Specific Blockers

### 3.1 Color Vocabulary Mismatch

**Verbeek et al. (2007)** -- *Learning Color Names from Real-World Images* (CVPR)
- Method: Probabilistic Latent Semantic Analysis (PLSA) learned from weakly labeled web images.
- Color space: L*a*b* (outperformed RGB and HSL).
- Key insight: Color names are distributions, not crisp regions. A pixel can have membership across multiple color name distributions.
- Applicability: HIGH. Replace hard HSV thresholds with probabilistic color name models in L*a*b* space.

**ABANICCO (2023)** -- *A New Color Space for Multi-Label Pixel Classification and Color Analysis* (Sensors)
- Method: Fuzzy color theory in CIELAB polar coordinates. Trapezoidal membership functions with gradients between adjacent colors.
- Key insight: Assigns multiple non-exclusive labels per pixel. Example: "86.09% Red and 13.91% Pink" or "58.77% Brown, 40.04% Red, and 1.19% Pink."
- Maps to ISCC-NBS color naming system (267 names, much richer than Berlin & Kay's 11 basic terms).
- Applicability: VERY HIGH. Ready-made architecture for our vocabulary mismatch problem.

**Fabric Color Naming (2018)** -- *Comparison of Fabric Color Naming Using RGB and HSV*
- Compared multiple color naming models with 7 distance metrics.
- Finding: No single color space wins universally. Low-saturation images work better with HSV + quadratic distance; medium with RGB + Euclidean; high with HSV + weighted Euclidean.
- Applicability: MEDIUM. Suggests adaptive color space selection based on image saturation.

### 3.2 Hymenophore / Gill Attachment

**Key finding: No published method explicitly classifies gill attachment from 2D photos.**

**Sulc et al. (WACV 2020)** -- *Fungi Recognition: A Practical Use Case* (FGVCx winner, 1,394 species)
- Method: End-to-end CNN (Inception-v4, Inception-ResNet-v2).
- Key finding: Grad-CAM shows models implicitly attend to gill regions, but never extract "gill attachment" as an explicit trait.
- Applicability: LOW for trait extraction. Confirms gills are visually important but offers no structured trait output method.

**Coprinoid Classification (2024)**
- Method: DPN, Xception, EfficientNet with Grad-CAM + Integrated Gradients.
- Key finding: High-performing models focus ~92% of attention on caps, gills, and stipes. Low-performing models drift to background (65-67%).
- Applicability: LOW for trait extraction. Reconfirms CNNs find gills discriminative but uses end-to-end classification.

**Conclusion:** Gill attachment classification from 2D images is an open problem in the literature. Attachment is a spatial/geometric property (relationship between gills and stem), not a texture pattern. Requires accurate stem mask, gill region, and geometric analysis of intersection.

### 3.3 2D Contour Shape Unreliability

**Key finding: The field has abandoned explicit shape descriptors for end-to-end learning.**

**Danish Fungi 2020 (DF20) / Picek et al. (2022)**
- CNN + Vision Transformer achieves 80.45% top-1 on 1,500+ species.
- Rich metadata: habitat, substrate, month, GPS location.
- Side-information fusion: Adding habitat + substrate + month + location improves accuracy by 2.95-4.8 pp and F1 by up to 7.1 points.
- Applicability: MEDIUM-HIGH. Validates hybrid pipeline design. Compensate for weak shape signals with stronger metadata fusion.

**Conclusion:** Single-view 2D shape classification is fundamentally ill-posed. Literature's solution is to not solve it explicitly. Pragmatic options: keep shape as low-confidence optional, use CNN's implicit shape knowledge, or rely on multi-view if available.

---

## 4. Recommendations

1. **Color:** Replace hard HSV binning with fuzzy L*a*b* multi-label color naming (ABANICCO-style). Map DB color terms to overlapping membership functions. This is the highest-impact, most actionable fix.

2. **Hymenophore:** Acknowledge as open problem. Either collect labeled training data for a dedicated attachment classifier, or treat gill attachment as user-input-only.

3. **Shape:** Do not invest in improving 2D contour shape extraction. Instead, strengthen non-visual priors (habitat, season, substrate) in the hybrid classifier, following Danish Fungi literature.
