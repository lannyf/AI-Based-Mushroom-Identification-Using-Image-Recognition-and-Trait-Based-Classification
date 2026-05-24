# Implementation Plan: Unified LLM-Centric Pipeline

## Overview
Refactor the mushroom identification system into a unified pipeline with:
- Two-photo input (above + below)
- YOLOv8 **3-class segmentation** (`cap`, `stem`, `underside`)
- **YOLO-driven case routing** — morphology determined by detected classes across both photos
- **CNN runs for every image** — always produces a prediction with an uncertainty flag; never omitted
- **CNN does not influence case routing** — only YOLO + two-photo check determines morphology
- **Multimodal LLM** as primary decision engine — initialized with `key.xml` tree knowledge, traverses tree internally, reasons over images + traits + CNN + trait database
- Agreement evaluation across CNN, Tree, and LLM

---

## Phase 1: Data Preparation (User) — COMPLETE

### 1.1 Download Images for 4 New Species (~2 hours) ✅
- Kungschampion (*Agaricus augustus* / AG.AU)
- Mandelriska (*Lactarius volemus* / LA.VO)
- Lakritsriska (*Lactarius helvus* / LA.HE)
- Blek fingersvamp (*Ramaria pallida* / RA.PA)
- Target: ~30 images each from iNaturalist + GBIF (CC-licensed)

### 1.2 Annotate Images in Roboflow (~3 hours with SAM) ✅

| Case | Images | YOLO Classes | Annotation Rules |
|------|--------|-------------|------------------|
| Classical | 75 | `cap`, `stem`, `underside` | Separate polygons per part per mushroom. Use above photo for cap, below for underside. |
| Puffball | 15 | `cap` only | Single polygon around whole round body. No stem/underside — case inferred from absence. |

**Total: 100 annotated images in `data/Manual_annotation/` (80 challenging + 20 basic cases).**

### 1.3 Export and Convert
1. Export from Roboflow as **COCO Segmentation** (3 classes)
2. Convert to YOLO format:
   ```bash
   python scripts/convert_coco_to_yolo.py \
       --coco-json "annotations.json" \
       --images-dir "images/" \
       --output-dir data/segmentation/yolo_3class \
       --rdp-epsilon 2.0
   ```

---

## Phase 2: Model Training (Colab)

### 2.1 Train YOLOv8n-seg (3-class)
- Upload dataset to Google Colab
- Train from `yolov8n-seg.pt` pretrained weights
- Hyperparameters: epochs=100, imgsz=640, batch=8, patience=20
- Expected time: 6–8 hours on T4 GPU
- **Decision gate (Day 3):** evaluate mAP per class.
  - `cap` mAP ≥ 0.60
  - `stem` mAP ≥ 0.50
  - `underside` mAP ≥ 0.45

### 2.2 CNN Retraining (13 species) — Stretch Goal
- Update `SPECIES_ID_TO_LABEL` with 13 species (10 ground-growing + 3 new)
- **Exclude tree-growing:** `fomitopsis_betulina` (BO.BA), `sparassis_crispa`
- Hyperparameters: epochs=20, batch=8, lr=3e-4
- Expected time: 2–3 hours on GPU
- **Fallback:** If time runs out or compute unavailable, keep existing 7-species CNN.
  - **Important:** The CNN still runs on every image regardless — just with 7 trained species.

---

## Phase 3: Multimodal LLM Upgrade

### 3.1 Upgrade from `llama3.2:3b` to Vision-Capable Model
Current `llama3.2:3b` is text-only. The LLM must process images directly.

| Model | Size | VRAM/RAM | Speed on CPU | Recommendation |
|-------|------|----------|-------------|----------------|
| `moondream` | 1.6B | ~2 GB | Fast | **Recommended** — smallest vision model, fast enough for thesis demo |
| `llava-phi3` | 3.8B | ~3 GB | Moderate | Good balance of quality and speed |
| `llava:7b` | 7B | ~5 GB | Slow | Better quality but may be too slow on CPU |
| `llama3.2-vision` | 11B | ~8 GB | Very slow | Too large for 13GB RAM + CPU |

**Recommended:** Start with `moondream` for speed, fall back to `llava-phi3` if visual reasoning quality is insufficient.

```bash
ollama pull moondream
# or
ollama pull llava-phi3
```

### 3.2 Image Input to LLM
The multimodal LLM receives the **raw photos** alongside the extracted traits:
```python
llm_prompt = {
    "images": [photo_above, photo_below],
    "extracted_traits": traits,
    "cnn_prediction": cnn_result,
    "tree_traversal": tree_result,
    "trait_db_matches": db_matches,
}
```

---

## Phase 4: Code Implementation (Me)

### 4.1 `models/mushroom_segmenter.py`
- Parse **3-class** YOLO output: `cap=0`, `stem=1`, `underside=2`
- Return `instances` list with class, mask, bbox, confidence per detection
- Aggregate detections across **both photos**:
  ```python
  instances_above = segment(photo_above)
  instances_below = segment(photo_below)
  all_instances = instances_above + instances_below
  ```
- Add `detect_case_from_yolo()` function — **class-driven, never uses CNN**:
  ```python
  def detect_case(all_instances):
      classes = {inst["class"] for inst in all_instances}
      
      elif "cap" in classes and ("stem" in classes or "underside" in classes):
          return "classical"
      elif "cap" in classes:
          return "puffball"  # cap only, no stem/underside in either photo
      else:
          return "uncertain"
  ```
- **Guard against cap-only misclassification:** The two-photo requirement is the primary guard. If the below photo reveals stem/underside, case is classical regardless of what the above photo shows.

### 4.2 `models/visual_trait_extractor.py`
- **Redesign:** Trait extractor receives `(image, yolo_instances)` only. CNN is **not** an input and does **not** influence case routing.
- Add `extract_traits(image, instances)` — dispatches by YOLO-detected case:

  **Puffball case (`cap` only in both photos):**
  ```python
  {
      "body_color": dominant_color(image, cap_mask),
      "body_shape": "round" if circularity(cap_mask) > 0.85 else "pear-shaped",
      "surface_texture": texture(image, cap_mask),
      "size_estimate": approximate_diameter(cap_mask),
      "morphology": "puffball"
  }
  ```

  }
  ```

  **Classical case (`cap` + `stem` + `underside`):**
  ```python
  {
      "cap": {
          "color": dominant_color(image, cap_mask),
          "shape": shape_descriptor(cap_mask),
          "surface_texture": texture(image, cap_mask)
      },
      "stem": {
          "color": dominant_color(image, stem_mask),
          "has_ring": detect_ring(image, stem_mask),
          "surface_texture": texture(image, stem_mask)
      },
      "underside": {
          "color": dominant_color(image, underside_mask),
          "type": "gills" if gill_pattern(underside_mask) else "pores",
          "spacing": gill_spacing(underside_mask)
      },
      "morphology": "classical"
  }
  ```

### 4.3 `models/cnn_classifier.py` — Always Runs
- **CNN is never optional.** It runs for every image.
- Output format:
  ```python
  {
      "species": "BO.ED",           # top-1 prediction, even if OOD
      "confidence": 0.67,           # raw softmax confidence
      "top_5": [...],               # full top-5 for LLM context
      "in_distribution": False,     # True if species in training set
      "conclusive": False,          # True if conf >= 0.40 and margin >= 0.15
      "uncertainty_reason": "species not in training set"  # or "low confidence", etc.
  }
  ```
- The LLM uses the full CNN output including uncertainty flags — it is never omitted.

### 4.4 `models/key_tree_traversal.py` — LLM-Internal Traversal
- **At startup:** Parse `key.xml` and inject the full decision tree structure into the LLM system prompt.
- The LLM receives tree structure + extracted traits and **traverses internally**:
  ```python
  system_prompt = """
  You are a mushroom identification assistant. You have access to the Swedish
  fungal key (key.xml) with the following decision tree structure:
  
  [Injected: full key.xml structure as nested JSON/text]
  
  When given visual traits, traverse the tree logically to reach a species
  or identify where the traits are ambiguous.
  """
  ```
- The traditional `_try_auto_answer()` method remains as a standalone engine for benchmark comparison, but the **LLM performs its own tree traversal** as part of reasoning.
- Tree result is passed to LLM as one evidence source among several.

### 4.5 `models/trait_database.py` (Comparator)
- Query the trait database for species matching the extracted features.
- Return top-N matches with similarity scores.
- Pass to LLM as `"trait_db_matches": [...]`.

### 4.6 `models/llm_classifier.py` — Multimodal Primary Engine
- **Model:** `moondream` or `llava-phi3` (multimodal)
- **Initialized at startup with:**
  1. `key.xml` decision tree structure (system prompt)
  2. Species taxonomy and morphological descriptions
- **Receives at inference time:**
  ```json
  {
    "images": [photo_above_base64, photo_below_base64],
    "morphology": "classical" | "puffball",
    "traits": { ... },
    "cnn_prediction": {
      "species": "BO.ED",
      "confidence": 0.67,
      "in_distribution": true,
      "conclusive": false,
      "uncertainty_reason": null
    },
    "tree_traversal": {
      "reached": "Boletus edulis",
      "path": ["..."],
      "stuck_at": null
    },
    "trait_db_matches": [
      {"species": "BO.ED", "score": 0.91},
      {"species": "BO.BA", "score": 0.78}
    ]
  }
  ```
- **LLM reasoning process:**
  1. Look at raw images → form visual impression
  2. Compare with extracted traits → validate consistency
  3. Perform internal tree traversal using traits → get tree species
  4. Compare with CNN prediction → note agreement or conflict
  5. Compare with trait database matches → cross-reference
  6. Synthesize final identification with confidence and reasoning

- **Output JSON:**
  ```json
  {
    "species": "Boletus edulis",
    "confidence": 0.85,
    "reasoning": "The cap is brown and convex, the underside shows pores (not gills), the stem is thick and white with a reticulated pattern. The CNN suggests BO.ED with moderate confidence. The tree traversal reaches Boletus edulis given these traits. The trait database strongly matches BO.ED.",
    "needs_clarification": false,
    "sources_agreement": "agree",
    "dissenting_sources": []
  }
  ```

### 4.7 `api/main.py`
- Add `/identify/unified` endpoint:
  ```python
  @app.post("/identify/unified")
  async def identify_unified(photo_above: UploadFile, photo_below: UploadFile):
      # 1. YOLOv8 on both photos (parallel)
      instances = segmenter.segment([photo_above, photo_below])
      
      # 2. Determine case from YOLO classes (CNN not involved)
      case = segmenter.detect_case(instances)
      
      # 3. Extract traits (CNN-independent)
      traits = trait_extractor.extract(photo_above, photo_below, instances, case)
      
      # 4. CNN prediction (always runs, even for OOD)
      cnn_result = cnn.predict(photo_above)
      
      # 5. Tree traversal (standalone engine for LLM context)
      tree_result = tree_engine.traverse(traits)
      
      # 6. Trait database lookup
      db_matches = trait_db.query(traits)
      
      # 7. Multimodal LLM prediction (primary decision)
      llm_result = llm.predict(
          images=[photo_above, photo_below],
          morphology=case,
          traits=traits,
          cnn_prediction=cnn_result,
          tree_traversal=tree_result,
          trait_db_matches=db_matches
      )
      
      # 8. Agreement evaluation
      agreement = evaluate_agreement(cnn_result, tree_result, llm_result)
      
      # 9. Return bundled response
      return {
          "llm": llm_result,
          "agreement": agreement,
          "traits": traits,
          "case": case,
          "cnn": cnn_result,
          "tree": tree_result,
          "db_matches": db_matches
      }
  ```

### 4.8 `api/schemas.py`
- Add `UnifiedIdentifyRequest` (two image fields)
- Add `UnifiedIdentifyResponse` (llm_result, agreement, traits, case, cnn, tree, db_matches)

### 4.9 Agreement Evaluator (revised)
```python
def evaluate_agreement(cnn_result, tree_result, llm_result):
    cnn_species = cnn_result["species"] if cnn_result["conclusive"] else None
    tree_species = tree_result.get("species") if tree_result else None
    llm_species = llm_result.get("species")
    
    sources = [("llm", llm_species)]
    if cnn_species:
        sources.append(("cnn", cnn_species))
    if tree_species:
        sources.append(("tree", tree_species))
    
    species_set = {sp for _, sp in sources}
    
    if len(species_set) == 1:
        return {"status": "agree", "species": llm_species, "sources": sources}
    elif len(species_set) == 2:
        return {"status": "disagree", "species": llm_species, "sources": sources}
    else:
        return {"status": "inconclusive", "species": llm_species, "sources": sources}
```

---

## Phase 5: Benchmark — Comparative Method Evaluation

The benchmark is designed to answer the thesis question: **Do standalone identification methods perform better separately or together?** It evaluates CNN, trait extractor, decision tree, and trait database both as independent classifiers and as components of the unified pipeline.

### 5.1 Benchmark Dataset

**Full evaluation set:** 60 images (5 per species × 12 species) from `data/raw/evaluation_images/`

**Confusing-case subsets** (to be decided when benchmark is implemented):

The user will select specific confusing mushroom pairs (e.g., lookalikes) to focus the analysis on cases where standalone methods are expected to struggle. The benchmark runner will support `--subset confusion` to evaluate only the selected pairs.

*Example candidate pairs (to be confirmed at implementation time):*
- Gulkantarell (*Cantharellus cibarius*) vs. Narrkantarell (*Hygrophoropsis aurantiaca*)
- Karl-Johan (*Boletus edulis*) vs. Brun sopp (*Boletus badius*)
- Flugsvamp (*Amanita muscaria*) vs. Vit flugsvamp (*Amanita virosa*)

### 5.2 Per-Sample Output Format

For every image, the benchmark produces a comparison row showing what **each method independently concluded** before the LLM synthesized anything:

```json
{
  "image": "evaluation_images/CA.CI/CA.CI_03.jpg",
  "ground_truth": "CA.CI",
  
  "cnn": {
    "prediction": "HY.PS",
    "confidence": 0.71,
    "in_distribution": true,
    "conclusive": true,
    "top_5": [["HY.PS", 0.71], ["CA.CI", 0.19], ...]
  },
  
  "trait_extractor": {
    "morphology": "classical",
    "cap": {"color": "yellow-orange", "shape": "funnel", "surface_texture": "smooth"},
    "stem": {"color": "yellow", "has_ring": false},
    "underside": {"color": "yellow", "type": "ridges", "forking": true}
  },
  
  "tree": {
    "prediction": "CA.CI",
    "conclusive": true,
    "path": ["Hur ser svampen ut? → åsar", "Vilken färg? → gul"],
    "auto_answered": 2,
    "stuck_at": null
  },
  
  "trait_db": {
    "top_match": "CA.CI",
    "top_score": 0.87,
    "rankings": [["CA.CI", 0.87], ["HY.PS", 0.82], ["CR.CO", 0.45], ...]
  },
  
  "unified": {
    "llm_prediction": "CA.CI",
    "llm_confidence": 0.91,
    "agreement_status": "disagree",
    "sources": {
      "cnn": "HY.PS",
      "tree": "CA.CI",
      "trait_db": "CA.CI",
      "llm": "CA.CI"
    },
    "reasoning": "CNN favours HY.PS due to colour similarity, but the trait database and tree both strongly point to CA.CI because the ridges are decurrent and forked — a hallmark of true chanterelles. The underside texture rule overrides the CNN's colour bias."
  }
}
```

### 5.3 Benchmark Report Structure

**CSV report** (`report.csv`) — one row per image:

| Column | Description |
|--------|-------------|
| `image` | Filename |
| `ground_truth` | True species ID |
| `cnn_pred` | CNN top-1 |
| `cnn_conf` | CNN confidence |
| `cnn_correct` | Boolean |
| `tree_pred` | Tree conclusion (or `"stuck"`) |
| `tree_correct` | Boolean |
| `tree_conclusive` | Boolean |
| `trait_db_top` | Database top match |
| `trait_db_score` | Match score |
| `trait_db_correct` | Boolean |
| `unified_pred` | LLM final prediction |
| `unified_conf` | LLM confidence |
| `unified_correct` | Boolean |
| `agreement` | `agree` / `disagree` / `partial` / `inconclusive` |
| `dissenters` | Which methods disagreed with unified result |

**Markdown report** (`report.md`) — thesis-ready tables:

1. **Overall Accuracy Comparison**
   | Method | Top-1 Accuracy | Coverage | Mean Time |

2. **Confusing-Pair Breakdown** (populated when pairs are selected)
   | Pair | CNN Acc | Tree Acc | Trait DB Acc | Unified Acc | Agreement Rate |

3. **Cases Where Unified Outperformed All Standalone Methods**
   | Image | CNN | Tree | DB | Unified | Why It Won |

4. **Cases Where Unified Was Wrong But a Standalone Method Was Right**
   | Image | CNN | Tree | DB | Unified | Failure Reason |

5. **Agreement Statistics**
   | Agreement Level | Count | % | Avg Unified Accuracy |

### 5.4 Metrics Computed

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | % of images where method's top prediction equals ground truth |
| **Top-3 Accuracy** | % where ground truth is in top-3 predictions |
| **Coverage** | % of images where method produced any prediction (tree may get stuck) |
| **Macro F1** | F1 averaged across all species classes |
| **Per-Pair Accuracy** | Accuracy on each confusing pair individually |
| **Agreement Rate** | % of images where ≥2 methods agree |
| **Unified Override Rate** | % of images where unified chose differently from the majority of standalone methods |

### 5.5 `benchmarks/runners/comparative_runner.py`

New runner that orchestrates all methods on each sample and collates results:

```python
class ComparativeRunner:
    def __init__(self):
        self.cnn = CNNRunner()
        self.tree = TreeRunner(mode="auto")
        self.trait_db = TraitDBRunner()
        self.unified = UnifiedRunner()  # calls /identify/unified
    
    def evaluate_sample(self, sample):
        return {
            "cnn": self.cnn.predict(sample),
            "tree": self.tree.predict(sample),
            "trait_db": self.trait_db.predict(sample),
            "unified": self.unified.predict(sample),
        }
```

### 5.6 Run Benchmark

```bash
# Full evaluation set (60 images)
python benchmarks/run_benchmark.py --mode comparative --subset all

# Confusing pairs only (pairs to be defined at implementation time)
python benchmarks/run_benchmark.py --mode comparative --subset confusion

# Specific species pair
python benchmarks/run_benchmark.py --mode comparative --species CA.CI HY.PS
```

### 5.7 Thesis Narrative

The benchmark is designed to produce evidence for these claims:

1. **CNN alone struggles on confusing pairs.** The CNN may achieve high overall accuracy but drop sharply on visually similar species because colour/shape features dominate.

2. **The trait extractor + tree excels on structure.** The decision tree can separate species by microscopic or structural traits (gill attachment, pore surface, ring presence) that CNNs may miss.

3. **The trait database captures nuanced similarity.** DB scores reflect genuine visual similarity between species; close scores indicate where human-like discrimination is required.

4. **Unified LLM synthesis outperforms any single method.** When sources disagree, the LLM weights structural and contextual evidence higher than superficial features.

5. **Agreement analysis reveals method strengths.** High agreement on easy species validates all methods; disagreement on hard species is where the unified pipeline adds value.

---

## Critical Path Timeline (1 Week)

| Day | Task | Owner | Hours |
|-----|------|-------|-------|
| 1 | Download new species images; annotate 105 images in Roboflow | User | 4–5 |
| 2 | Export COCO; upload to Colab; start YOLOv8 3-class training | User | 2 + overnight |
| 3 | Evaluate YOLOv8 mAP; **Go/No-Go decision** on 3-class | User | 2 |
| 4 | Pull multimodal LLM; inject key.xml; implement segmenter + trait extractor | Me | 6–8 |
| 5 | Implement unified endpoint + LLM prompt + internal tree traversal | Me | 6–8 |
| 6 | Benchmark + debug | Both | 4–6 |
| 7 | Thesis writing | User | 4–6 |

---

## Go/No-Go Decision Gate (Day 3)

**Continue with 3-class if:**
- `cap` mAP ≥ 0.60
- `stem` mAP ≥ 0.50
- `underside` mAP ≥ 0.45

**Fallback if any class below threshold:**
- Use single-class YOLOv8 for detection
- Case routing falls back to heuristic mask analysis + two-photo check
- Trait extractor still works
- Multimodal LLM still runs with raw images
- CNN still runs for every image
- Thesis still demonstrates the architecture

---

## Files to Modify (Summary)

| File | Change |
|------|--------|
| `api/main.py` | Add `/identify/unified`, agreement evaluator |
| `api/schemas.py` | Add unified request/response schemas |
| `models/mushroom_segmenter.py` | Parse **3-class** output, class-driven case detection (no CNN) |
| `models/visual_trait_extractor.py` | Part-specific extraction, **CNN-independent**, case routing |
| `models/cnn_classifier.py` | Always runs; return prediction + uncertainty flags for all images |
| `models/key_tree_traversal.py` | Export tree structure for LLM injection; keep standalone engine |
| `models/trait_database.py` | Query and return top-N matches for LLM context |
| `models/llm_classifier.py` | **Multimodal** prompt; initialized with key.xml; synthesizes all sources |
| `benchmarks/runners/comparative_runner.py` | New benchmark runner |

---

## Signal Flow Diagram

```
Two Photos
    │
    ├──→ YOLOv8 Segmentation (3-class)
    │         │
    │         ↓
    │    Case Router (class-driven, both photos)
    │    ├─ cap+stem+underside  → classical traits
    │    ├─ cap only (both)     → puffball traits
    │    └─ uncertain           → heuristic fallback
    │         │
    │         ↓
    │    Trait Extractor (CNN-independent)
    │         │
    │         ↓
    │    Trait Database Query
    │         │
    ├──→ CNN (always runs) ──────┤
    │    returns prediction +    │
    │    uncertainty flags       │
    │    (even for OOD)          │
    │         │                  │
    ├──→ Key Tree Engine ────────┤
    │    (standalone traversal   │
    │     for LLM context)       │
    │         │                  │
    │         ↓                  ↓
    │    Multimodal LLM (primary decision)
    │    ├─ Initialized with key.xml at startup
    │    ├─ Receives raw images
    │    ├─ Receives traits + CNN + tree + DB matches
    │    ├─ Performs internal tree traversal
    │    └─ Synthesizes final identification
    │         │
    │         ↓
    │    Agreement Evaluator
    │         │
    │         ↓
    │    Final Response
```

---

## Notes

- **CNN always runs.** Even for OOD species, it returns a prediction with `in_distribution: false` and an uncertainty reason. The LLM sees this and down-weights the CNN accordingly.
- **CNN never influences case routing.** Puffball vs classical ambiguity is resolved by the two-photo check only.
- **LLM is multimodal.** It sees the raw images directly, not just extracted traits. This allows it to catch trait extraction errors and use visual intuition.
- **LLM knows the tree.** The full `key.xml` structure is injected into the system prompt at startup. The LLM traverses it logically as part of reasoning.
- **Tree-growing exclusions:** `fomitopsis_betulina` and `sparassis_crispa` excluded from CNN training. YOLO still detects them; LLM can identify them from raw images + traits.
- All old endpoints remain untouched as fallbacks.
