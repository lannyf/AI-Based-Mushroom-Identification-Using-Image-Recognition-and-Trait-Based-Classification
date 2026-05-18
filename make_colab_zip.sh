#!/usr/bin/env bash
# Create a minimal ZIP for Google Colab benchmarking
# Excludes training data, SDKs, build artifacts, and other large files

set -euo pipefail

ZIP_PATH="${1:-$HOME/mushroom-benchmark.zip}"

# Detect project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating Colab benchmark ZIP at: $ZIP_PATH"
echo "Excluding training images, SDKs, build artifacts, etc."
echo ""

zip -r "$ZIP_PATH" . \
  -x ".venv/*" \
  -x ".git/*" \
  -x "__pycache__/*" \
  -x "*.pyc" \
  -x ".dart_tool/*" \
  -x ".kimi/*" \
  -x ".cache/*" \
  -x "flutter/*" \
  -x "data/raw/images/*" \
  -x "data/Yolov8/train/*" \
  -x "data/Yolov8/valid/*" \
  -x "data/Yolov8/test/*" \
  -x "data/Yolov8/*.jpg" \
  -x "data/Yolov8/*.png" \
  -x "data/SegMask/*" \
  -x "data/SegMaskJS/*" \
  -x "data/segmentation.zip" \
  -x "artifacts/yolov8_seg_runs/*" \
  -x "artifacts/cnn_training_history.json" \
  -x "mushroom_id_app/*" \
  -x "java-backend/*"

SIZE_MB=$(du -m "$ZIP_PATH" | cut -f1)
echo ""
echo "✅ Created: $ZIP_PATH (${SIZE_MB} MB)"

if [ "$SIZE_MB" -gt 200 ]; then
    echo "⚠️  Warning: ZIP is over 200 MB. Browser upload may be unreliable."
    echo "   Consider using Google Drive + gdown (see Google colab/GOOGLE_COLAB_BENCHMARK.md)"
    echo ""
    echo "   Large directories in project:"
    du -sh * | sort -hr | head -10
fi
