# video-eval

AI-generated video quality evaluation framework with plugin architecture, multi-dimension scoring, and VLM-powered analysis.

## Key Features

- **Plugin Architecture** — Evaluators, extractors, backends, and fusion strategies are all pluggable via entry-points
- **Multi-Dimension Scoring** — Technical quality, AIGC defects, product fidelity, compliance, and VLM-based subjective judgment
- **VLM-Powered** — Integrates Gemini, OpenAI, and local Qwen3-VL for intelligent video understanding
- **Cross-Device** — Auto-detects CUDA / Apple MPS / CPU; develop on Mac, deploy on GPU with zero config changes
- **Structured Output** — Each dimension produces score + evidence + reasoning + suggestion, ready for downstream systems
- **Batch Processing** — Directory scan or CSV manifest for bulk evaluation

## Quick Start

```bash
pip install video-eval

# Evaluate a single video
video-eval eval --video demo.mp4 --video-type general

# Check device capabilities
video-eval device info
```

## Installation

### Basic (CPU, mock backend)

```bash
pip install video-eval
```

### With optional features

```bash
pip install 'video-eval[asr]'    # ASR extraction (faster-whisper)
pip install 'video-eval[ocr]'    # OCR extraction (easyocr)
pip install 'video-eval[gpu]'    # GPU support (torch + open-clip)
pip install 'video-eval[all]'    # Everything
```

### From source

```bash
git clone https://github.com/anthropics/video-eval.git
cd video-eval
pip install -e '.[dev]'
```

See [docs/getting-started.md](docs/getting-started.md) for Docker and detailed setup instructions.

## CLI Usage

### Single video evaluation

```bash
video-eval eval \
  --video product_hero.mp4 \
  --video-type main_image \
  --product-title "Wireless Earbuds Pro" \
  --selling-points "Active noise cancellation" \
  --selling-points "40h battery life" \
  --output report.json
```

### Feature extraction only

```bash
video-eval extract --video demo.mp4 --output features.json
```

### Batch evaluation

```bash
# From directory (all .mp4/.avi/.mov/.mkv/.webm/.flv files)
video-eval batch --input ./videos/ --output results.json

# From CSV manifest
video-eval batch --input manifest.csv --output-format jsonl --output results.jsonl
```

### Plugin management

```bash
video-eval plugins                        # List all registered plugins
video-eval plugins --type evaluator       # Filter by type
video-eval plugins --detail vlm_judge     # Show plugin details
video-eval plugins --available-only       # Only device-compatible plugins
```

### Configuration

```bash
video-eval config check                   # Validate current config
video-eval config check --config my.yaml  # Validate specific file
```

### Device info

```bash
video-eval device info                    # Show device, memory, backend status
```

### Config overrides via --set

```bash
video-eval eval --video demo.mp4 --video-type general \
  --set evaluators.vlm_judge.backend=api \
  --set backends.api.provider=openai
```

## Configuration

video-eval uses a three-layer configuration system (defaults < config file < CLI overrides).

Copy the example config to get started:

```bash
cp config.yaml.example config.yaml
```

See [docs/configuration.md](docs/configuration.md) for the full reference.

## Architecture

```
┌─────────┐     ┌────────────┐     ┌────────────┐     ┌────────┐     ┌────────┐
│  Input  │────▶│ Extractors │────▶│ Evaluators │────▶│ Fusion │────▶│ Output │
│ (video) │     │            │     │            │     │        │     │(report)│
└─────────┘     └────────────┘     └────────────┘     └────────┘     └────────┘
                 video_meta          technical_quality   weighted_veto   JSON
                 asr                  aigc_defect                        JSONL
                 ocr                  product_fidelity
                 clip_features        compliance
                                     vlm_judge ──▶ Backend (mock/api/local)
```

All components are discovered via the **Registry** system, supporting both built-in plugins and third-party packages installed via pip entry-points.

See [docs/architecture.md](docs/architecture.md) for contributor-level details.

## Plugin Development

Create custom evaluators, backends, or extractors and distribute them as pip packages:

```python
from video_eval.core.base import BaseEvaluator
from video_eval.core.registry import register_evaluator

@register_evaluator("my_custom_eval")
class MyCustomEvaluator(BaseEvaluator):
    name = "my_custom_eval"
    ...
```

See [docs/plugin-development.md](docs/plugin-development.md) for the full guide.

## Documentation

- [Getting Started](docs/getting-started.md) — Installation, Docker, first evaluation
- [Configuration Reference](docs/configuration.md) — All config options explained
- [Plugin Development](docs/plugin-development.md) — Build and publish custom plugins
- [Architecture](docs/architecture.md) — Internals for contributors
- [Contributing](CONTRIBUTING.md) — How to contribute
- [Changelog](CHANGELOG.md) — Release history

## License

[Apache-2.0](LICENSE)
