# Project Insight: Cap Shape Analysis Segmentation Limitation

**Date:** 2026-05-07
**Source:** `models/visual_trait_extractor.py`, functions `analyse_shape()` and `analyse_shape_masked()`
**Context:** Thesis Method section review — shape analysis critique

---

## Problem Statement

The `analyse_shape()` function claims to classify **cap shape** (convex, flat, funnel-shaped, bell-shaped, wavy), but it does not actually isolate the cap. Instead, it operates on whichever blob the thresholding or segmentation produces — typically the **entire mushroom silhouette** (cap + stem) or, in the unmasked case, the largest bright object in the full image (which may include background).

This creates a fundamental measurement error: the **stem distorts the aspect ratio and circularity** that are supposed to describe only the cap.

---

## Current Implementation

### Unmasked version (`analyse_shape`, line 136)
- Thresholds the full image with Otsu
- Finds the largest contour
- That contour may be: mushroom + stem + grass + leaves + shadows

### Masked version (`analyse_shape_masked`, line 307)
- Uses the YOLO segmentation mask to remove background
- BUT the mask still contains **cap + stem + gills as one blob**
- No separation of cap from stem

**Result:** A side-photographed mushroom with a visible stem will have:
- **True cap aspect ratio** (top-down): ~1.0 (round)
- **Measured aspect ratio** (side view, cap+stem): ~0.4 (tall) → misclassified as "bell-shaped" or "funnel-shaped"

---

## Why It Matters for the Key Tree

The Swedish `key.xml` asks part-specific questions that the current system cannot answer:
- `"Vilken färg har skivorna?"` (What colour are the gills?) — not handled
- `"Vilken färg har hatten?"` (What colour is the cap?) — uses whole-mushroom dominant colour as proxy
- `"Vilken färg har saften?"` (What colour is the latex?) — physically impossible from photo

Without cap-specific segmentation, the auto-answer system cannot reliably map visual traits to these questions.

---

## Potential Solutions

### Option 1: Multi-Class YOLO Segmentation (Architecturally Correct, Labour-Intensive)

Train YOLOv8 with separate classes:
- `cap`
- `stem`
- `gills` / `pores` / `ridges`
- `ring` (annulus)

**Advantages:**
- Clean isolation of each morphological part
- Enables true cap shape analysis
- Enables part-specific colour analysis (cap colour vs stem colour vs gill colour)
- Would close the auto-answer gaps in the key tree traversal

**Disadvantages:**
- Requires pixel-level masks for 4–5 classes on thousands of images
- Severe class imbalance (stems are thin, gills are often occluded)
- High species variability (chanterelles have ridges, boletes have pores, milkcaps have gills)
- Some fungi have no stem (trumpet fungi, bracket fungi, coral fungi)

**Verdict:** The technically superior solution, but likely overkill for a thesis timeline.

### Option 2: Geometric Heuristics on Existing Mask (Simple, Fast)

Keep the single-class "mushroom" mask, but estimate the cap region:

```python
def extract_cap_from_mask(mask):
    """Approximate cap as the upper, wider portion of the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    # Assume cap occupies the top 60% of the bounding box
    cap_region = mask[y:y + int(h * 0.6), x:x + w]
    return cap_region
```

Or fit an ellipse and take the upper half:
```python
ellipse = cv2.fitEllipse(cnt)
# Cap ≈ region above the ellipse centreline
```

**Advantages:**
- No retraining needed
- Works with existing YOLO model immediately
- Fast to implement and test

**Disadvantages:**
- Assumes "cap is top, stem is bottom" — fails for side-angle photos
- Fails for stemless fungi (trumpets, brackets, coral fungi)
- Still an approximation

### Option 3: Hybrid Approach (Recommended for Future Work)

Use the existing mask but apply different geometric rules depending on preliminary classification:

```python
if preliminary_shape in {"funnel-shaped", "trumpet"}:
    # No clear cap/stem boundary; analyse whole mask
    region = mask
elif aspect_ratio < 0.8:
    # Tall profile; cap is probably the top 50%
    region = top_half_of(mask)
else:
    # Round, convex cap dominates the silhouette
    region = mask
```

**Advantages:**
- Adapts to morphology without requiring new training data
- Could be implemented as a post-processing layer on existing masks

**Disadvantages:**
- Adds complexity without guarantees
- Still fundamentally heuristic-based

---

## Empirical Evidence

Real-image testing on Fly Agaric (`Amanita muscaria`) and Chanterelle (`Cantharellus cibarius`) images revealed:
- The unmasked `analyse_shape` is dominated by background tones
- The masked version improves background removal but still includes the stem
- No ground-truth validation exists in the test suite; `TestAnalyseShape` only tests on synthetic white circles

---

## Thesis Framing

**As a limitation:**
> *"The cap shape classifier operates on the largest thresholded contour of the full image, or on the YOLO segmentation mask when available. In neither case is the cap isolated from the stem; the computed aspect ratio and circularity reflect the overall mushroom silhouette and are sensitive to photography angle. This is a known limitation that affects species with prominent stems photographed from the side."*

**As future work:**
> *"Future work could employ multi-class instance segmentation (cap, stem, gills) to enable part-specific trait extraction. Alternatively, a geometric post-processing step on the existing whole-mushroom mask could estimate the cap boundary by exploiting the typical 'wider-at-top' morphology of agarics and boletes."*

---

## Related Files

- `models/visual_trait_extractor.py` — `analyse_shape()` (line 136), `analyse_shape_masked()` (line 307)
- `models/key_tree_traversal.py` — `_try_auto_answer()` (line 236)
- `data/raw/key.xml` — Part-specific colour questions that cannot be auto-answered
- `tests/test_visual_trait_extractor.py` — Only synthetic shape tests (no real-image validation)
