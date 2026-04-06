# Data Dictionary - Mushroom Identification Dataset

## Overview

This document describes all data files and their schemas for the mushroom identification system. The dataset contains structured information about 50 mushroom species, including taxonomic data, morphological traits, images, and lookalike relationships.

---

## File: `species.csv`

**Purpose:** Master record of all mushroom species in the system.

**Location:** `data/raw/species.csv`

### Columns

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `species_id` | String (5 chars) | Yes | Unique identifier for the species. Format: 2-letter genus abbreviation + `.` + 2-letter species abbreviation (e.g. `CA.CI` = *Cantharellus cibarius*). | `CA.CI`, `AM.PH`, `GY.ES` |
| `scientific_name` | String | Yes | Full scientific binomial nomenclature. | `Cantharellus cibarius` |
| `swedish_name` | String | Yes | Common name in Swedish. | `Kantarell` |
| `english_name` | String | Yes | Common name in English. | `Chanterelle` |
| `edible` | Boolean | Yes | Whether the species is safe to eat. `TRUE` or `FALSE`. | `TRUE`, `FALSE` |
| `toxicity_level` | Enum | Yes | Toxicity rating. Values: `SAFE`, `TOXIC`, `EXTREMELY_TOXIC`. | `SAFE`, `EXTREMELY_TOXIC` |
| `priority_lookalike` | String | Yes | ID of the primary dangerous lookalike species. Use `NONE` if no lookalike. | `HY.PS`, `AM.PH`, `NONE` |

### Constraints

- `species_id` must be unique
- `species_id` must not be null
- `edible=FALSE` requires `toxicity_level` to be `TOXIC` or `EXTREMELY_TOXIC`
- `edible=TRUE` requires `toxicity_level` to be `SAFE`
- `priority_lookalike` must reference a valid `species_id` or be `NONE`

### Example Rows

```csv
CA.CI,Cantharellus cibarius,Kantarell,Chanterelle,TRUE,SAFE,HY.PS
AM.PH,Amanita phalloides,Dödskallesvamp,Death Cap,FALSE,EXTREMELY_TOXIC,NONE
```

---

## File: `species_traits.csv`

**Purpose:** Detailed morphological traits for each species organized by category.

**Location:** `data/raw/species_traits.csv`

### Columns

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `species_id` | String | Yes | Reference to `species.csv`. Foreign key. | `CA.CI` |
| `trait_category` | Enum | Yes | Morphological category. Values: `CAP`, `GILLS`, `STEM`, `FLESH`, `HABITAT`, `SEASON`, `GROWTH`. | `CAP`, `GILLS` |
| `trait_name` | String | Yes | Specific trait within category. | `shape`, `color`, `attachment` |
| `trait_value` | String | Yes | The actual value for this trait. Can be single value or pipe-separated alternatives. | `funnel-shaped`, `yellow\|orange`, `3-7` |
| `value_type` | Enum | Yes | Data type of the value. Values: `categorical`, `numeric`, `range`, `text`. | `categorical`, `range` |
| `variability` | String | No | Notes on how this trait varies (by age, season, etc.). | `varies with age`, `consistent`, `yellow to deep orange` |

### Trait Categories & Names

#### CAP (Hatt)
- `shape`: funnel-shaped, convex, flat, hemispherical, conical, etc.
- `color`: white, cream, yellow, orange, brown, red, purple, gray, black, multicolor
- `surface_texture`: smooth, wrinkled, cracked, scaly, sticky, matte, shiny
- `size_cm`: numeric range (e.g., "3-7")
- `margin`: smooth, wavy, grooved, torn, rolled, inrolled

#### GILLS (Lameller)
- `attachment`: free, adnate, sinuate, decurrent, pores, unequal
- `density`: crowded, moderately spaced, distant, small (for pores)
- `color`: white, cream, yellow, pink, red, orange, brown, gray, black, variable
- `edge`: smooth, serrated, irregular, blunt

#### STEM (Stam)
- `shape`: cylindrical, bulbous, tapered, curved, equal
- `color`: white, cream, yellow, orange, brown, red, purple, gray
- `surface`: smooth, ridged, fibrous, scaly, powdery, reticulate
- `hollow_solid`: solid, stuffed, hollow
- `size_cm`: numeric range
- `ring`: present, absent, faint

#### FLESH (Kött)
- `color`: white, cream, yellow, orange, pink, red, brown, gray
- `texture`: firm, soft, brittle, waxy, corky, dry
- `smell`: not distinctive, pleasant, unpleasant, acrid, fishy, fruity, radish
- `bruising`: no change, blue, red/pink, yellow/orange, black

#### HABITAT (Växtplats)
- `substrate_type`: soil, wood, dung, moss, leaf litter
- `tree_association`: oak, beech, birch, aspen, pine, spruce, fir, mixed, none
- `elevation`: lowland, mid-elevation, mountain, coastal, widespread

#### SEASON (Säsong)
- `season`: Values like "april|may", "july|august|september"

#### GROWTH (Växtmönster)
- `pattern`: solitary, scattered, gregarious, clustered, fairy rings, caespitose

### Value Format Specifications

- **Categorical values:** Lowercase, hyphen-separated (e.g., `funnel-shaped`, `pale-yellow`)
- **Range values:** "min-max" format (e.g., `3-7` cm, `8-15` cm)
- **Multiple values:** Pipe-separated (e.g., `oak|birch|spruce`, `july|august|september`)
- **Text values:** Descriptive text (e.g., "fruity pleasant smell")

### Constraints

- `species_id` must reference valid entry in `species.csv`
- Each `(species_id, trait_category, trait_name)` combination must be unique
- `value_type` must match the format of `trait_value`

### Example Rows

```csv
CA.CI,CAP,shape,funnel-shaped,categorical,consistent
CA.CI,CAP,color,yellow-orange,categorical,yellow to deep orange
CA.CI,CAP,size_cm,3-7,range,varies with age
BO.ED,STEM,color,white|brown,categorical,white with brown network
AM.PH,FLESH,smell,radish,categorical,distinctive radish smell
```

---

## File: `species_images.csv`

**Purpose:** Metadata for all images in the dataset.

**Location:** `data/raw/species_images.csv`

### Columns

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `image_id` | String | Yes | Unique identifier for the image. Format: `IMG_<species_id>_<number>`. | `IMG_CA.CI_001`, `IMG_AM.PH_005` |
| `species_id` | String | Yes | Reference to `species.csv`. | `CA.CI` |
| `file_path` | String | Yes | Relative path to image file from `data/raw/`. | `images/CA.CI/CA.CI_001_young_sunny_top.jpg` |
| `image_stage` | Enum | Yes | Growth stage of mushroom in image. Values: `young`, `developing`, `mature`. | `young`, `mature` |
| `lighting` | Enum | Yes | Lighting conditions. Values: `direct_sunlight`, `dappled`, `shade`, `artificial`. | `dappled`, `artificial` |
| `angle` | Enum | Yes | Camera angle/perspective. Values: `top-down`, `side`, `ground-level`, `close-up`. | `top-down`, `close-up` |
| `source` | Enum | Yes | Image source. Values: `field_guide`, `user_photo`, `online_database`, `mushroom_db`. | `field_guide`, `user_photo` |
| `quality` | Enum | Yes | Image quality rating. Values: `LOW`, `MEDIUM`, `HIGH`. | `HIGH` |
| `suitable_for_training` | Boolean | Yes | Whether image is suitable for ML training. `TRUE` or `FALSE`. | `TRUE`, `FALSE` |

### Naming Convention for Image Files

File names follow the pattern: `<species_id>_<sequence>_<stage>_<lighting>_<angle>.jpg`

Examples:
- `CA.CI_001_young_sunny_top.jpg` - Chanterelle, image 1, young stage, sunny lighting, top-down angle
- `AM.PH_003_mature_artificial_closeup.jpg` - Death cap, image 3, mature, artificial lighting, close-up
- `BO.ED_002_developing_shade_side.jpg` - Porcini, image 2, developing, shaded, side angle

### Constraints

- `image_id` must be unique
- `species_id` must reference valid entry in `species.csv`
- `file_path` must be relative to `data/raw/`
- Images with `suitable_for_training=FALSE` will be excluded from model training
- Minimum 10 suitable images per species recommended

### Image Requirements

- **Resolution:** Minimum 800x600 pixels, target 1200+ pixels
- **Format:** JPG, PNG with 3-channel RGB
- **Clarity:** Sharp, in-focus, identifiable features visible
- **Lighting:** Natural or controlled, no extreme shadows
- **Diversity:** Multiple angles, growth stages, and seasonal variations

### Example Rows

```csv
IMG_CA.CI_001,CA.CI,images/CA.CI/CA.CI_001_young_sunny_top.jpg,young,direct_sunlight,top-down,user_photo,HIGH,TRUE
IMG_CA.CI_005,CA.CI,images/CA.CI/CA.CI_005_habitat_context.jpg,mature,natural,ground-level,user_photo,MEDIUM,TRUE
IMG_AM.PH_001,AM.PH,images/AM.PH/AM.PH_001_young_stage.jpg,young,natural,top-down,field_guide,HIGH,TRUE
```

---

## File: `lookalikes.csv`

**Purpose:** Document dangerous lookalike relationships between species.

**Location:** `data/raw/lookalikes.csv`

### Columns

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `lookalike_id` | String | Yes | Unique identifier. Format: `LA<number>`. | `LA001`, `LA008` |
| `edible_species_id` | String | Yes | Reference to edible species (should have `edible=TRUE`). | `CA.CI`, `BO.ED` |
| `toxic_species_id` | String | Yes | Reference to toxic lookalike species (should have `edible=FALSE`). | `HY.PS`, `AM.PH` |
| `confusion_likelihood` | Enum | Yes | How often these are confused. Values: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`. | `HIGH`, `CRITICAL` |
| `distinguishing_features` | String | Yes | Detailed text describing key differences for identification. | "Chanterelle: thick meaty ridges..." |

### Constraints

- `lookalike_id` must be unique
- Both species IDs must reference valid entries in `species.csv`
- `edible_species_id` must have `edible=TRUE`
- `toxic_species_id` must have `edible=FALSE`
- Each pair should only appear once (no duplicates)

### Distinguishing Features Format

Write detailed, practical descriptions that a user can use in the field:
- Compare cap appearance (size, color, shape, surface)
- Compare gill/pore structure (attachment, spacing, color)
- Compare stem characteristics (color, shape, markings)
- Compare flesh characteristics (color, smell, bruising)
- Highlight the most distinctive differences first

Example:
```
True chanterelle: thick meaty ridges (false gills). False chanterelle: thin blade-like true gills. 
Chanterelle: flesh-colored interior. False: watery fragile. 
Chanterelle: fruity pleasant smell. False: no distinctive smell.
Chanterelle stem: solid. False stem: hollow fragile.
```

### Example Rows

```csv
LA001,CH001,FALSE_CH,HIGH,"True chanterelle: thick meaty ridges. False: thin blade-like gills. Chanterelle: flesh-colored firm. False: watery fragile..."
LA006,HO001,FUNERAL_BELL,CRITICAL,"Honey mushroom: 2-6cm yellow-brown clustered. Funeral bell: 1-3cm brown scattered. Honey: white fragrant. Funeral: thin dark..."
```

---

## File: `dataset_split.csv`

**Purpose:** Assign images to train/validation/test sets for evaluation.

**Location:** `data/raw/dataset_split.csv`

### Columns

| Column | Type | Required | Description | Example |
|--------|------|----------|-------------|---------|
| `species_id` | String | Yes | Reference to `species.csv`. | `CH001` |
| `image_id` | String | Yes | Reference to `species_images.csv`. | `IMG_CH001_001` |
| `split_set` | Enum | Yes | Dataset partition. Values: `TRAIN`, `VALIDATION`, `TEST`. | `TRAIN`, `TEST` |
| `reason` | String | No | Why assigned to this set (for documentation). | `balanced_distribution`, `lookalike_training`, `held_out_for_evaluation` |

### Constraints

- Both `species_id` and `image_id` must reference valid entries
- Images with `suitable_for_training=FALSE` should not be in TRAIN set
- Avoid including multiple photos of same specimen in different sets
- Each image should appear only once in splits

### Distribution Guidelines

- **TRAIN:** 70% of images - used for model training
- **VALIDATION:** 15% of images - used for hyperparameter tuning
- **TEST:** 15% of images - held out for final evaluation

### Stratification Strategy

- Ensure all species appear in TRAIN set
- Aim for balanced species representation across splits
- Separate images of same specimen across splits when possible
- Balance by growth stage and condition across splits

### Example Rows

```csv
CH001,IMG_CH001_001,TRAIN,balanced_distribution
CH001,IMG_CH001_004,VALIDATION,robust_evaluation
CH001,IMG_CH001_005,TEST,held_out_for_evaluation
DEATH,IMG_DEATH_001,TRAIN,dangerous_species_training
```

---

## Directory Structure: `data/raw/images/`

### Organization

```
data/raw/images/
├── CH001/           # Chanterelle
│   ├── CH001_001_young_sunny_top.jpg
│   ├── CH001_002_mature_shade_side.jpg
│   └── ...
├── BU001/           # Porcini
│   ├── BU001_001_young_dappled_top.jpg
│   └── ...
├── MO001/           # Morel
├── ... (17 more species directories)
└── FUNERAL_BELL/    # Funeral Bell
```

### Image File Naming

Files in each species folder follow the pattern:
`<species_id>_<sequence>_<stage>_<lighting>_<angle>.jpg`

Where:
- `<species_id>`: 3-6 character species code (e.g., CH001, DEATH)
- `<sequence>`: 001-999 (incremental number for ordering)
- `<stage>`: young, developing, mature
- `<lighting>`: sunny, dappled, shade, artificial
- `<angle>`: top, side, ground, closeup

---

## Processed Data Output Formats

### `data/processed/dataset.json`

JSON export of all raw data for programmatic access.

```json
{
  "species": [...],
  "traits": [...],
  "images": [...],
  "lookalikes": [...],
  "splits": [...]
}
```

### `data/processed/trait_features.csv`

Wide-format feature matrix with one row per species and trait values as columns.

```
species_id,CAP_shape,CAP_color,CAP_size_cm,...
CH001,funnel-shaped,yellow-orange,3-7,...
BU001,convex|flat,brown,5-30,...
```

### `data/processed/training_metadata.json`

Metadata for training including image processor settings and augmentation info.

```json
{
  "train": [...],
  "test": [...],
  "species": [...],
  "image_processor": {
    "target_size": [224, 224],
    "normalization": "ImageNet"
  },
  "augmentation": "enabled"
}
```

---

## Data Quality Standards

### Completeness

- **Target:** 95%+ trait data per species
- **Target:** Minimum 10 images per species (preferably 15-20)
- **Missing values:** Document with NULL and explain in validation report

### Consistency

- **Categorical values:** Must match approved vocabulary
- **Ranges:** Must be "min-max" format with numeric values
- **Species references:** All foreign keys must point to existing entries

### Validation Checks

Run the validation script regularly:
```bash
python data/validate_data.py --data-dir data/raw --stats
```

This validates:
- ✓ All required columns present
- ✓ No duplicate species IDs
- ✓ Lookalike species references valid
- ✓ Split sets properly distributed (70/15/15)
- ✓ Minimum image counts per species
- ✓ Trait coverage completeness

---

## Data Collection Instructions

### For New Species

1. Create new `species_id` (3-6 chars + numbers)
2. Add species to `species.csv` with all required fields
3. Extract traits from field guide or sources → add to `species_traits.csv`
4. Create species image folder: `data/raw/images/<species_id>/`
5. Collect/obtain 10+ images → add to `species_images.csv`
6. If applicable, add lookalike relationships → `lookalikes.csv`
7. Assign images to splits → `dataset_split.csv`
8. Run validation: `python data/validate_data.py`

### Image Sourcing

Preferred sources (in order):
1. Nya Svampboken (if digitized)
2. Licensed mushroom photography collections
3. MushroomObserver (iNaturalist alternative)
4. Arkivoc or Swedish mushroom databases
5. User-contributed field observations

**Always verify:** Image quality, species identification accuracy, and usage rights.

---

## References

- **Field Guide:** Holmberg, P., & Marklund, H. (2019). *Nya Svampboken*. Bonnier Fakta.
- **Trait Standards:** Traditional mycological identification keys and field guide conventions
- **Data Organization:** Inspired by ImageNet, COCO, and other large-scale ML datasets
- **Image Formats:** Following standard computer vision dataset practices (JPG/PNG, 224x224 recommended for CNNs)

---

**Version:** 1.0  
**Last Updated:** 2026-03-14  
**Data Responsibility:** Project team
