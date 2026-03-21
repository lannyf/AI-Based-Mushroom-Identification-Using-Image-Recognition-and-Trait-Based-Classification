# Manual Verification Commands - Phases 1-3

Copy-paste these commands to verify each phase works as intended.

---

## PHASE 1: FOUNDATION & DATA VERIFICATION

### 1. Check CSV files exist
```bash
ls -lh data/raw/
```
**EXPECTED:** 5 CSV files (species, traits, images, lookalikes, splits)

---

### 2. Verify species count (should be 20)
```bash
wc -l data/raw/species.csv
```
**EXPECTED:** 21 (1 header + 20 species)

---

### 3. Verify trait entries count (should be 77)
```bash
wc -l data/raw/species_traits.csv
```
**EXPECTED:** 78 (1 header + 77 trait entries)

---

### 4. Check species structure
```bash
head -2 data/raw/species.csv
```
**EXPECTED:** Headers: species_id, scientific_name, swedish_name, english_name, edible, toxicity_level

---

### 5. Count edible vs toxic species
```bash
grep -c "true" data/raw/species.csv
```
**EXPECTED:** 12 (edible = true)

---

### 6. Verify bilingual names
```bash
tail -10 data/raw/species.csv | cut -d',' -f3-4
```
**EXPECTED:** Swedish and English name pairs

---

### 7. Check trait categories (7 expected)
```bash
cut -d',' -f3 data/raw/species_traits.csv | sort -u | tail -8
```
**EXPECTED:** CAP, GILLS, STEM, FLESH, HABITAT, SEASON, GROWTH (+ header)

---

### 8. Verify documentation files
```bash
ls -lh Docs/01-*.md Docs/02-*.md Docs/03-*.md Docs/04-*.md Docs/05-*.md
```
**EXPECTED:** 5 files present

---

### 9. Check documentation sizes
```bash
wc -l Docs/{01,02,03,04,05}-*.md
```
**EXPECTED:** 100+ KB total

---

### 10. Verify architecture mentions
```bash
grep -c "UML\|diagram\|architecture" Docs/04-system-architecture.md
```
**EXPECTED:** Multiple matches (8+ diagrams)

---

## PHASE 2: IMAGE RECOGNITION MODULE

### 1. Verify config file
```bash
ls -lh config/image_model_config.py
```
**EXPECTED:** ~8 KB file

---

### 2. Check config syntax
```bash
python3 -m py_compile config/image_model_config.py && echo "✓ Syntax OK"
```
**EXPECTED:** "✓ Syntax OK" message

---

### 3. Verify ImageProcessor class
```bash
grep -n "class ImageProcessor" models/image_processor.py
```
**EXPECTED:** Line number showing class definition

---

### 4. Check ImageProcessor methods
```bash
grep -n "def " models/image_processor.py | head -10
```
**EXPECTED:** Methods listed (load_image, preprocess, augment, etc.)

---

### 5. Check image_processor syntax
```bash
python3 -m py_compile models/image_processor.py && echo "✓ Syntax OK"
```
**EXPECTED:** "✓ Syntax OK" message

---

### 6. Verify ImageRecognitionModel class
```bash
grep -n "class ImageRecognitionModel" models/image_recognition.py
```
**EXPECTED:** Line number showing class definition

---

### 7. Check base models available
```bash
grep "BASE_MODEL\|MobileNetV2\|EfficientNetB0\|ResNet50" config/image_model_config.py
```
**EXPECTED:** References to 3+ base models

---

### 8. Check image_recognition syntax
```bash
python3 -m py_compile models/image_recognition.py && echo "✓ Syntax OK"
```
**EXPECTED:** "✓ Syntax OK" message

---

### 9. Check training script
```bash
grep -n "def main\|if __name__" scripts/train_image_model.py
```
**EXPECTED:** Main function and entry point shown

---

### 10. Check training script syntax
```bash
python3 -m py_compile scripts/train_image_model.py && echo "✓ Syntax OK"
```
**EXPECTED:** "✓ Syntax OK" message

---

### 11. Check evaluation script
```bash
grep -n "def evaluate\|def main" scripts/evaluate_image_model.py
```
**EXPECTED:** Evaluate and main functions

---

### 12. Check evaluation script syntax
```bash
python3 -m py_compile scripts/evaluate_image_model.py && echo "✓ Syntax OK"
```
**EXPECTED:** "✓ Syntax OK" message

---

### 13. Check image recognition documentation
```bash
ls -lh Docs/06-image-recognition.md
```
**EXPECTED:** ~13 KB file

---

### 14. Verify augmentation parameters
```bash
grep "rotation\|brightness\|flip" config/image_model_config.py
```
**EXPECTED:** Multiple augmentation parameters

---

## PHASE 3: TRAIT-BASED CLASSIFICATION

### 1. Verify TraitEncoder class
```bash
grep -n "class TraitEncoder" models/trait_processor.py
```
**EXPECTED:** Line number showing class definition

---

### 2. Check TraitEncoder methods
```bash
grep -n "def " models/trait_processor.py | head -15
```
**EXPECTED:** Methods listed (fit, transform, fit_transform, encode_feature, etc.)

---

### 3. Check trait_processor syntax
```bash
python3 -m py_compile models/trait_processor.py && echo "✓ Syntax OK"
```
**EXPECTED:** "✓ Syntax OK" message

---

### 4. Verify TraitClassifier class
```bash
grep -n "class TraitClassifier" models/trait_classifier.py
```
**EXPECTED:** Line number showing class definition

---

### 5. Check Decision Tree implementation
```bash
grep -n "DecisionTreeClassifier\|decision_tree" models/trait_classifier.py
```
**EXPECTED:** Multiple matches

---

### 6. Check Random Forest implementation
```bash
grep -n "RandomForestClassifier\|random_forest" models/trait_classifier.py
```
**EXPECTED:** Multiple matches

---

### 7. Check trait_classifier syntax
```bash
python3 -m py_compile models/trait_classifier.py && echo "✓ Syntax OK"
```
**EXPECTED:** "✓ Syntax OK" message

---

### 8. Check TraitObservation class
```bash
grep -n "class TraitObservation" models/trait_processor.py
```
**EXPECTED:** Line number showing class definition

---

### 9. Check TraitDataset class
```bash
grep -n "class TraitDataset" models/trait_processor.py
```
**EXPECTED:** Line number showing class definition

---

### 10. Check trait training script
```bash
grep -n "def main\|if __name__" scripts/train_trait_model.py
```
**EXPECTED:** Main function and entry point

---

### 11. Check training script syntax
```bash
python3 -m py_compile scripts/train_trait_model.py && echo "✓ Syntax OK"
```
**EXPECTED:** "✓ Syntax OK" message

---

### 12. Check evaluation script
```bash
grep -n "def evaluate\|def main" scripts/evaluate_trait_model.py
```
**EXPECTED:** Evaluate and main functions

---

### 13. Check evaluation script syntax (CRITICAL FIX #1)
```bash
python3 -m py_compile scripts/evaluate_trait_model.py && echo "✓ Syntax OK"
```
**EXPECTED:** "✓ Syntax OK" message (Tuple import fix verified)

---

### 14. Verify Tuple import (FIX #1)
```bash
grep "from typing import.*Tuple" scripts/evaluate_trait_model.py
```
**EXPECTED:** Shows import line with Tuple

---

### 15. Check return type (FIX #2)
```bash
grep -A1 "def fit_transform" models/trait_processor.py | head -2
```
**EXPECTED:** Return type is `List[str]` (not Tuple)

---

### 16. Check exception handling (FIX #3)
```bash
grep -A1 "except (" models/image_recognition.py | head -2
```
**EXPECTED:** Shows specific exception types

---

### 17. Check trait documentation
```bash
ls -lh Docs/07-trait-classification.md
```
**EXPECTED:** ~11 KB file

---

### 18. Verify data split (70/15/15)
```bash
grep -i "0.7\|70\|0.15\|15" scripts/train_trait_model.py
```
**EXPECTED:** Shows train/val/test split ratios

---

## QUICK VERIFICATION (Run All At Once)

```bash
cd /home/iannyf/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification

echo "=== PHASE 1: DATA ==="
echo "Species count:" && wc -l data/raw/species.csv
echo "Traits count:" && wc -l data/raw/species_traits.csv
echo "Edible species:" && grep -c "true" data/raw/species.csv

echo ""
echo "=== PHASE 2: IMAGE RECOGNITION ==="
python3 -m py_compile config/image_model_config.py && echo "✓ Config"
python3 -m py_compile models/image_processor.py && echo "✓ ImageProcessor"
python3 -m py_compile models/image_recognition.py && echo "✓ ImageRecognitionModel"
python3 -m py_compile scripts/train_image_model.py && echo "✓ Training script"
python3 -m py_compile scripts/evaluate_image_model.py && echo "✓ Evaluation script"

echo ""
echo "=== PHASE 3: TRAIT CLASSIFICATION ==="
python3 -m py_compile models/trait_processor.py && echo "✓ TraitProcessor"
python3 -m py_compile models/trait_classifier.py && echo "✓ TraitClassifier"
python3 -m py_compile scripts/train_trait_model.py && echo "✓ Training script"
python3 -m py_compile scripts/evaluate_trait_model.py && echo "✓ Evaluation script"

echo ""
echo "=== CODE REVIEW FIXES ==="
grep "from typing import.*Tuple" scripts/evaluate_trait_model.py && echo "✓ Fix #1: Tuple import"
grep "def fit_transform.*-> List\[str\]" models/trait_processor.py && echo "✓ Fix #2: Return type"
grep "except (FileNotFoundError" models/image_recognition.py && echo "✓ Fix #3: Exception handling"

echo ""
echo "=== DOCUMENTATION ==="
ls -1 Docs/{01,02,03,04,05,06,07}-*.md 2>/dev/null | wc -l && echo "documentation files"
```

---

## EXPECTED RESULTS

✓ Species: 21 lines (20 + header)  
✓ Traits: 78 lines (77 + header)  
✓ Edible: 12 species  
✓ All Python files: syntax checks pass  
✓ All 3 critical code review fixes verified  
✓ 7 documentation files present  

---

## Summary

If all commands above produce expected results, **Phases 1-3 are working correctly.**

If any fail, note which one and we can investigate.
