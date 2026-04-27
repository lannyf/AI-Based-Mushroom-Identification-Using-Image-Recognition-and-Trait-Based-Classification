"""
Configuration for the live mushroom image recognition pipeline.

This is the single source of truth for:
  - model architecture (timm backbone, input size, species list)
  - image preprocessing (normalisation, resize/crop policy)
  - training hyperparameters
  - artifact paths

Both inference (models/cnn_classifier.py) and training (scripts/train_cnn.py)
import from this module so the two stay in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

for _dir in (ARTIFACTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# timm model identifier — must be a valid timm.create_model() name
BASE_MODEL: str = "efficientnet_b3"

# Input size handed to the model after all preprocessing (height, width)
INPUT_SIZE: Tuple[int, int] = (300, 300)

# Resize dimension used before center-crop during inference/validation
RESIZE_SIZE: int = 320

# Species handled by the classifier — order matters for the model output logits
SPECIES: List[str] = [
    "Fly Agaric",
    "Chanterelle",
    "False Chanterelle",
    "Porcini",
    "Other Boletus",
    "Amanita virosa",
    "Black Trumpet",
]

NUM_CLASSES: int = len(SPECIES)

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

# ImageNet normalization statistics (RGB)
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# ============================================================================
# TRAINING
# ============================================================================

DEFAULT_EPOCHS: int = 20
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_LEARNING_RATE: float = 3e-4

VAL_FRACTION: float = 0.2
EARLY_STOPPING_PATIENCE: int = 5

# Phase 1: train only the classification head
HEAD_ONLY_EPOCHS_FRACTION: float = 1 / 3
HEAD_ONLY_LR_FACTOR: float = 1.0

# Phase 2: fine-tune full network at reduced LR
FINETUNE_LR_FACTOR: float = 0.1

# Data augmentation (training only)
TRAIN_AUGMENTATION: Dict[str, object] = {
    "random_resized_crop_scale": (0.7, 1.0),
    "color_jitter": (0.3, 0.3, 0.2),  # brightness, contrast, saturation
    "random_rotation": 20,  # degrees
    "random_horizontal_flip": True,
}

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

WEIGHTS_FILENAME: str = "cnn_weights.pt"
WEIGHTS_PATH: Path = ARTIFACTS_DIR / WEIGHTS_FILENAME

HISTORY_FILENAME: str = "cnn_training_history.json"
HISTORY_PATH: Path = ARTIFACTS_DIR / HISTORY_FILENAME

# ============================================================================
# INFERENCE
# ============================================================================

# Device preference — "cuda" falls back to CPU automatically when unavailable
DEVICE_PREFERENCE: str = "cuda"

# Confidence threshold below which predictions are considered uncertain
CONFIDENCE_THRESHOLD: float = 0.5

# Number of top predictions to return
TOP_K: int = 5

# ============================================================================
# HELPERS
# ============================================================================


def get_config_dict() -> Dict[str, object]:
    """Return all core configuration as a dictionary."""
    return {
        "base_model": BASE_MODEL,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "species": SPECIES,
        "weights_path": str(WEIGHTS_PATH),
        "default_epochs": DEFAULT_EPOCHS,
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "default_learning_rate": DEFAULT_LEARNING_RATE,
        "device_preference": DEVICE_PREFERENCE,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(get_config_dict(), indent=2))
