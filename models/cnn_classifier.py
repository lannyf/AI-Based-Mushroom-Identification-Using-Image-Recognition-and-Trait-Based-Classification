"""
CNN-based mushroom species classifier using EfficientNet-B3 via timm.

Two operating modes:
  1. Fine-tuned  — loads weights from artifacts/cnn_weights.pt after training
                   with scripts/train_cnn.py. Gives real ML predictions.
  2. Untrained   — no weights file found; returns None so the caller can fall
                   back to the classical CV-based scorer. The model is still
                   built (verifying timm is available) but not used for scoring.

Species handled (mirrors TARGET_SPECIES in api/main.py):
  Fly Agaric, Chanterelle, False Chanterelle, Porcini,
  Other Boletus, Amanita virosa, Black Trumpet
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Species registry — must match api/main.py TARGET_SPECIES order
# ---------------------------------------------------------------------------

SPECIES: List[str] = [
    "Fly Agaric",
    "Chanterelle",
    "False Chanterelle",
    "Porcini",
    "Other Boletus",
    "Amanita virosa",
    "Black Trumpet",
]

NUM_CLASSES = len(SPECIES)

# Default weights file location
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = _PROJECT_ROOT / "artifacts" / "cnn_weights.pt"


class MushroomCNN:
    """
    EfficientNet-B3 classifier for the 7 target mushroom species.

    Usage
    -----
    cnn = MushroomCNN()              # auto-loads weights if present
    scores = cnn.predict(img_bytes)  # returns None when untrained
    """

    def __init__(self, weights_path: Path = WEIGHTS_PATH) -> None:
        self._model = None
        self._transform = None
        self._trained = False
        self._device = "cpu"

        try:
            self._build(weights_path)
        except ImportError as exc:
            logger.warning(
                "timm/torch not installed — CNN classifier disabled. "
                "Run: pip install timm torch\n%s", exc
            )
        except Exception as exc:
            logger.warning("CNN classifier could not be initialised: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, weights_path: Path) -> None:
        """Build the EfficientNet-B3 model and optionally load weights."""
        import torch
        import timm
        from torchvision import transforms

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build EfficientNet-B3 with our custom head (7 classes)
        # pretrained=True downloads ImageNet-21k weights on first run
        self._model = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=NUM_CLASSES,
        )
        self._model.eval()
        self._model.to(self._device)

        # Standard ImageNet preprocessing (timm models expect this)
        self._transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if weights_path.exists():
            checkpoint = torch.load(str(weights_path), map_location=self._device)
            # Support both raw state_dict and checkpoint dicts
            state = checkpoint.get("model_state_dict", checkpoint)
            self._model.load_state_dict(state)
            self._trained = True
            logger.info("CNN: loaded fine-tuned weights from %s", weights_path)
        else:
            logger.info(
                "CNN: no fine-tuned weights at %s — predictions disabled "
                "(run scripts/train_cnn.py once training images are collected)",
                weights_path,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True only when fine-tuned weights have been loaded."""
        return self._trained

    def predict(self, image_bytes: bytes) -> Optional[Dict[str, float]]:
        """
        Run inference on raw image bytes.

        Returns
        -------
        dict mapping species name → normalised confidence (0-1), or
        None if the model is not available / not fine-tuned.
        """
        if self._model is None or not self._trained:
            return None

        try:
            import torch
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self._transform(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

            return {sp: float(p) for sp, p in zip(SPECIES, probs)}

        except Exception as exc:
            logger.error("CNN inference failed: %s", exc)
            return None

    def top_k(self, image_bytes: bytes, k: int = 5) -> Optional[List[Tuple[str, float]]]:
        """Return sorted list of (species, confidence) for top-k species."""
        scores = self.predict(image_bytes)
        if scores is None:
            return None
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# Module-level singleton — imported by visual_trait_extractor
_instance: Optional[MushroomCNN] = None


def get_classifier() -> MushroomCNN:
    """Return the shared MushroomCNN singleton (lazy init)."""
    global _instance
    if _instance is None:
        _instance = MushroomCNN()
    return _instance
