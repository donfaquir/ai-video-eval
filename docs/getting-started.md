# Getting Started

## Prerequisites

- **Python 3.11+**
- **ffmpeg** — Required for video frame extraction and metadata reading

Install ffmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (scoop)
scoop install ffmpeg
```

## Installation

### pip (recommended)

```bash
# Core only (CPU, mock backend)
pip install video-eval

# With ASR support
pip install 'video-eval[asr]'

# With OCR support
pip install 'video-eval[ocr]'

# With GPU support (PyTorch + OpenCLIP)
pip install 'video-eval[gpu]'

# Everything
pip install 'video-eval[all]'
```

### From source

```bash
git clone https://github.com/anthropics/video-eval.git
cd video-eval
pip install -e '.[dev]'
```

### Docker

#### CPU image (API backends)

```bash
docker build -f docker/Dockerfile.cpu -t video-eval:cpu .
docker run --rm -v $(pwd)/data:/data video-eval:cpu \
  eval --video /data/demo.mp4 --video-type general
```

#### GPU image (local VLM)

```bash
docker build -f docker/Dockerfile.gpu -t video-eval:gpu .
docker run --rm --gpus all -v $(pwd)/data:/data video-eval:gpu \
  eval --video /data/demo.mp4 --video-type general \
  --set evaluators.vlm_judge.backend=local
```

#### Docker Compose

```bash
docker compose up
```

The compose file mounts `./data`, `./config.yaml`, and `./results` by default.

## Verify Installation

```bash
# Check CLI is accessible
video-eval --version

# Check device detection
video-eval device info

# Validate config
video-eval config check

# List available plugins
video-eval plugins
```

Expected output from `video-eval device info`:

```
Device Information
========================================
  Device type:   cpu
  Is GPU:        False
  Total memory:  16.00 GB
  Free memory:   8.42 GB
  PyTorch:       not installed

Backend Status
----------------------------------------
  faster-whisper: not installed
  open-clip:      not installed
```

## First Evaluation

### 1. Prepare a config file

```bash
cp config.yaml.example config.yaml
```

The default config uses the `mock` backend, which returns deterministic scores without requiring any API keys or GPU.

### 2. Run a basic evaluation

```bash
video-eval eval --video sample.mp4 --video-type general
```

This runs all enabled evaluators and outputs a JSON report to stdout:

```json
{
  "video_path": "sample.mp4",
  "video_type": "general",
  "overall_score": 0.72,
  "grade": "B",
  "passed": true,
  "veto_reasons": [],
  "dimension_results": {
    "technical_quality": { "score": 0.85, "status": "scored", ... },
    "aigc_defect": { "score": 0.70, "status": "scored", ... },
    "cross_modal": { "score": 0.60, "status": "scored", ... }
  },
  "suggestions": [...],
  "meta": { ... }
}
```

### 3. Evaluate with product context (main_image type)

```bash
video-eval eval \
  --video product_hero.mp4 \
  --video-type main_image \
  --product-title "Wireless Earbuds Pro" \
  --selling-points "Active noise cancellation" \
  --selling-points "40h battery life" \
  --product-images reference.jpg \
  --output report.json
```

### 4. Switch to a real VLM backend

Set your API key and update the backend:

```bash
export GEMINI_API_KEY="your-key-here"

video-eval eval \
  --video demo.mp4 \
  --video-type general \
  --set evaluators.vlm_judge.backend=api \
  --set backends.api.provider=gemini
```

### 5. Batch evaluation

Create a CSV manifest (`manifest.csv`):

```csv
video_path,video_type,product_title,selling_points
./videos/hero1.mp4,main_image,Product A,Feature 1|Feature 2
./videos/hero2.mp4,main_image,Product B,Feature 3
./videos/ad1.mp4,external,,
```

Run batch:

```bash
video-eval batch --input manifest.csv --output results.json
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Grade A or B (passed) |
| 1 | Grade C (not passed, but no hard failure) |
| 2 | Configuration error |
| 3 | Grade REJECT (veto triggered) |
| 4 | Evaluator error (at least one dimension failed) |
| 5 | No dimensions scored |
| 6 | Extraction error (required extractor failed) |

## Next Steps

- [Configuration Reference](configuration.md) — Tune weights, thresholds, and evaluator settings
- [Plugin Development](plugin-development.md) — Build custom evaluators
- [Architecture](architecture.md) — Understand the internals
