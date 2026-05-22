#!/usr/bin/env python3
"""Build a clean zip for Google Colab benchmark runs.

Excludes heavy directories (.venv, .git, training data, Flutter app, etc.)
and ensures the YOLO weights symlink exists.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "mushroom-benchmark.zip"

# Directories to include entirely
INCLUDE_DIRS = [
    "benchmarks",
    "models",
    "api",
    "config",
    "data/raw/Benchmark",
]

# Individual files to include (relative to project root)
INCLUDE_FILES = [
    "data/dataset_utils.py",
    "data/raw/key.xml",
    "data/raw/species_traits.xml",
    "data/raw/species.csv",
    "data/raw/lookalikes.csv",
    "data/Yolov8/best(1).pt",
    "data/Yolov8/best.pt",          # symlink to best(1).pt
    "artifacts/cnn_weights.pt",
    "benchmarks/evaluation_manifest.csv",
    "benchmarks/evaluation_manifest_v2.csv",
    "requirements.txt",
    "Makefile",
]

# Patterns to exclude even inside included dirs
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "*.egg-info",
]


def should_exclude(path: Path) -> bool:
    parts = path.parts
    name = path.name
    for pat in EXCLUDE_PATTERNS:
        if pat.startswith("*") and name.endswith(pat.lstrip("*")):
            return True
        if pat in parts:
            return True
    return False


def main():
    # Ensure symlink exists
    best_pt = PROJECT_ROOT / "data" / "Yolov8" / "best.pt"
    best1_pt = PROJECT_ROOT / "data" / "Yolov8" / "best(1).pt"
    if not best_pt.exists():
        if best1_pt.exists():
            best_pt.symlink_to(best1_pt.name)
            print(f"Created symlink: {best_pt} -> {best1_pt.name}")
        else:
            raise FileNotFoundError(f"YOLO weights missing: {best1_pt}")

    # Build file list
    files_to_add: set[Path] = set()

    for rel_dir in INCLUDE_DIRS:
        src_dir = PROJECT_ROOT / rel_dir
        if not src_dir.exists():
            print(f"Warning: directory not found: {src_dir}")
            continue
        for root, _, files in os.walk(src_dir):
            for f in files:
                p = Path(root) / f
                if should_exclude(p):
                    continue
                files_to_add.add(p)

    for rel_file in INCLUDE_FILES:
        p = PROJECT_ROOT / rel_file
        if not p.exists():
            print(f"Warning: file not found: {p}")
            continue
        files_to_add.add(p)

    # Write zip
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        print(f"Removed old zip: {ZIP_PATH}")

    total_size = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files_to_add):
            arcname = p.relative_to(PROJECT_ROOT)
            # Resolve symlinks so target content is stored, not the link
            real_path = p.resolve()
            zf.write(real_path, arcname)
            total_size += real_path.stat().st_size

    zip_size = ZIP_PATH.stat().st_size
    print(f"\nCreated: {ZIP_PATH}")
    print(f"  Files: {len(files_to_add)}")
    print(f"  Uncompressed: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Compressed:   {zip_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
