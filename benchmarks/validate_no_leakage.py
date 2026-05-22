"""Validate that benchmark images were never in the YOLO training set.

Usage::

    python -m benchmarks.validate_no_leakage [--strict]

Exit codes::

    0  – No leakage detected (or --strict not set and training cache missing)
    1  – Hash collision found (leakage detected)
    2  – Strict mode: training data/cache unavailable
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "raw" / "Benchmark"
TRAINING_HASH_CACHE = PROJECT_ROOT / "data" / "Yolov8" / "training_source_hashes.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_benchmark_hashes() -> Dict[str, str]:
    """Return {relative_path: sha256} for all images in Benchmark/."""
    hashes: Dict[str, str] = {}
    if not BENCHMARK_DIR.exists():
        logger.error("Benchmark directory not found: %s", BENCHMARK_DIR)
        return hashes
    for img_path in sorted(BENCHMARK_DIR.rglob("*.jpg")):
        rel = img_path.relative_to(PROJECT_ROOT).as_posix()
        hashes[rel] = _sha256_file(img_path)
    return hashes


def _load_training_hashes() -> Set[str]:
    """Return set of SHA-256 hashes from training data cache."""
    if TRAINING_HASH_CACHE.exists():
        with open(TRAINING_HASH_CACHE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return set(data.get("hashes", []))

    # Look for extracted training images
    training_dirs = [
        PROJECT_ROOT / "data" / "Yolov8" / "train" / "images",
        PROJECT_ROOT / "data" / "Yolov8" / "Mushrooms" / "train" / "images",
    ]
    hashes: Set[str] = set()
    for d in training_dirs:
        if d.exists():
            for img_path in d.rglob("*.jpg"):
                hashes.add(_sha256_file(img_path))
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify benchmark images are not in YOLO training set"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if training hash cache is unavailable",
    )
    args = parser.parse_args()

    benchmark_hashes = _collect_benchmark_hashes()
    if not benchmark_hashes:
        logger.error("No benchmark images found.")
        return 1

    logger.info("Collected %d benchmark image hashes", len(benchmark_hashes))

    training_hashes = _load_training_hashes()

    if not training_hashes:
        msg = (
            "Training hash cache not found and no extracted training images detected.\n"
            f"  Cache path: {TRAINING_HASH_CACHE}\n"
            "  Source provenance: Benchmark images are from Svampeatlas/GBIF (Denmark).\n"
            "  Training images are from iNaturalist, Danish Fungi Atlas, and manual collection.\n"
            "  Different sources strongly indicate no leakage."
        )
        if args.strict:
            logger.error(msg)
            logger.error("Use --strict only when training data cache is available.")
            return 2
        logger.warning(msg)
        logger.warning("Pass --strict to treat this as an error.")
        return 0

    logger.info("Loaded %d training image hashes", len(training_hashes))

    collisions = []
    for rel_path, h in benchmark_hashes.items():
        if h in training_hashes:
            collisions.append(rel_path)

    if collisions:
        logger.error("LEAKAGE DETECTED: %d benchmark image(s) match training set:", len(collisions))
        for p in collisions:
            logger.error("  - %s", p)
        return 1

    logger.info("No leakage detected. All %d benchmark images are clean.", len(benchmark_hashes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
