# Google Colab Benchmark Guide

Run the full comparative benchmark (System A + System B) on Google Colab with GPU acceleration.

---

## Prerequisites

1. A Google account with access to [Google Colab](https://colab.research.google.com)
2. A `mushroom-benchmark.zip` file created from your project (see Step 1 below)
3. The Colab notebook `Mushroom_Benchmark_Colab.ipynb`

---

## Step 1: Create a Minimal ZIP (Local Machine)

The benchmark only needs code, model weights, and 57 benchmark images. Exclude everything else:

```bash
cd ~/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification

zip -r ~/mushroom-benchmark.zip . \
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
```

**Expected size: ~100–150 MB**

If your zip is still over 500 MB, check what's bloating it:

```bash
cd ~/projekt/AI-Based-Mushroom-Identification-Using-Image-Recognition-and-Trait-Based-Classification
du -sh * | sort -hr | head -15
```

---

## Step 2: Upload to Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook**
3. Select `Mushroom_Benchmark_Colab.ipynb` from your project
4. Drag `mushroom-benchmark.zip` into the **file browser panel on the left** (or use the upload cell in the notebook)

---

## Step 3: Configure Runtime (Critical)

1. **Runtime → Change runtime type**
2. Select **T4 GPU** from the Hardware accelerator dropdown
3. Click **Save**

---

## Step 4: Run All Cells

**Runtime → Run all** (or press `Ctrl+F9`)

The notebook will execute sequentially:

| Cell | Purpose | Est. Time |
|------|---------|-----------|
| 1 | Upload ZIP (if not dragged) | 1–2 min |
| 2 | Extract project zip | 10–30 s |
| 3 | Install Python dependencies | 2–3 min |
| 4 | Verify GPU | 5 s |
| 5 | Install Ollama | 30–60 s |
| 6 | Start Ollama server | 5–10 s |
| 7 | Pull `gemma3:4b` model | 3–5 min |
| 8 | Verify setup | 5 s |
| 9 | Run benchmark | **2–4 hours** |
| 10 | Download results | 10–30 s |

---

## Step 5: Keep Colab Alive

Colab free tier disconnects idle sessions after ~90 minutes. To prevent this during the long benchmark run:

1. Open browser **Developer Tools** (`F12` or `Ctrl+Shift+I`)
2. Go to the **Console** tab
3. Paste and press Enter:

```javascript
function ConnectButton(){
    console.log("Keeping Colab alive...");
    document.querySelector("colab-connect-button").click();
}
setInterval(ConnectButton, 60000);
```

This clicks the connect button every 60 seconds.

---

## What the Benchmark Does

### System A — Standalone Methods
Each method operates independently:

- **CNN**: Single-image classification with EfficientNet-B3
- **Tree**: YOLO segmentation → trait extraction → programmatic tree traversal (`KeyTreeEngine`)
- **DB**: Trait extraction → database ranking of all 57 species
- **LLM**: Raw vision LLM (`gemma3:4b`) with both above/below images, no extra context

### System B — Unified LLM Synthesis
The LLM aggregates all System A signals into a single prediction:

1. YOLO segmentation + trait extraction
2. CNN prediction
3. **LLM navigates the tree** (one-shot): receives tree structure + traits + CNN hint + images, outputs its chosen path
4. Path validator checks against `key.xml`
5. Database comparison using validated (or partial) tree conclusion
6. Final LLM synthesis of all signals

---

## Output Files

After completion, the notebook downloads `benchmark_results.zip` containing:

```
report.json    # Full structured data
report.csv     # One row per specimen, one column per method
report.md      # Thesis-ready Markdown report
```

The Markdown report separates System A and System B clearly:

```markdown
## 1. System A — Standalone Methods
## 2. System B — Unified LLM Synthesis
## 3. Raw Accuracy Difference: System B vs System A
## 4. Confusing-Pair Breakdown
## 5. Cases Where System B Outperformed All System A Methods
## 6. Cases Where System B Was Wrong But a System A Method Was Right
## 7. Agreement Statistics
## 8. Accuracy by Scenario
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ZIP upload fails / truncated | Browser upload is unreliable for 700+ MB. Use `gdown` from Google Drive instead (see Alternative below) |
| `zip not found` | Make sure `mushroom-benchmark.zip` is in `/content/`, not a subfolder |
| Ollama connection refused | Restart the Ollama server cell; check `!curl http://localhost:11434` |
| Out of disk space | `gemma3:4b` is ~3.3 GB. Check: `!df -h`. Free space with `!pip cache purge` |
| Session disconnected mid-run | Results are lost on disconnect. Consider running in smaller batches |
| CUDA/GPU not used by Ollama | Ollama auto-detects GPU. Check `!nvidia-smi` shows a T4 |
| Benchmark too slow | Ensure T4 GPU is selected. CPU-only inference is ~5× slower |

---

## Alternative: Upload via Google Drive (`gdown`)

If direct upload keeps failing for large files, upload to Drive and use `gdown`:

```python
# In Colab, run this instead of the upload cell
!pip install -q gdown

# Replace FILE_ID with your Google Drive file ID
!gdown --id FILE_ID -O /content/mushroom-benchmark.zip

import os
size_mb = os.path.getsize('/content/mushroom-benchmark.zip') / (1024*1024)
print(f"Downloaded: {size_mb:.1f} MB")
```

---

## Alternative: Run Without Notebook

If you prefer raw commands in a Colab cell:

```python
# After extracting project
%cd /content/mushroom-project

import os
os.environ["OLLAMA_TIMEOUT"] = "600"
os.environ["OLLAMA_NUM_PREDICT"] = "512"

!python3 -m benchmarks.run_comparative \
    --manifest benchmarks/evaluation_manifest.csv \
    --output-dir artifacts/benchmarks/colab_run \
    --methods all
```
