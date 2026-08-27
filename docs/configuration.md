# Configuration Reference

## Three-Layer Priority

Configuration is resolved in this order (later overrides earlier):

1. **Built-in defaults** — Hardcoded in each plugin's `config_schema`
2. **Config file** — `config.yaml` in the working directory, or specified via `--config`
3. **CLI overrides** — `--set key=value` flags

## Config File Location

The ConfigLoader searches in order:

1. Path specified by `--config` CLI flag
2. `./config.yaml` in the current directory
3. Built-in defaults (no file needed)

## Full Configuration Sections

### `device`

Controls hardware device selection.

```yaml
device:
  preferred: auto  # auto | cuda | mps | cpu
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `preferred` | string | `auto` | Device preference. `auto` picks best available (cuda > mps > cpu). |

### `extractors`

Global extraction settings and per-extractor configuration.

```yaml
extractors:
  fps: 1                # Frame sampling rate (frames per second)
  max_frames: 64        # Maximum frames to extract

  asr:
    enabled: true
    model_size: large-v3   # faster-whisper model size
    language: auto         # auto-detect or ISO code (e.g., "zh", "en")
    beam_size: 5

  ocr:
    confidence: 0.5        # Minimum confidence threshold

  clip_features:
    model_name: ViT-SO400M-14-SigLIP-384
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fps` | int | `1` | Frames sampled per second of video |
| `max_frames` | int | `64` | Cap on total extracted frames |
| `asr.enabled` | bool | `true` | Enable ASR extraction (requires `[asr]` extra) |
| `asr.model_size` | string | `large-v3` | Whisper model variant |
| `asr.language` | string | `auto` | Force language or auto-detect |
| `asr.beam_size` | int | `5` | Beam search width |
| `ocr.confidence` | float | `0.5` | Discard OCR detections below this score |
| `clip_features.model_name` | string | `ViT-SO400M-14-SigLIP-384` | OpenCLIP model identifier |

### `backends`

VLM backend configurations. Each backend type has its own section.

```yaml
backends:
  local:
    model: Qwen/Qwen3-VL-8B-Instruct

  api:
    provider: gemini       # gemini | openai
    model: gemini-3-flash
    timeout: 30
    max_retries: 3
    retry_base: 1.0

  mock: {}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `local.model` | string | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID for local inference |
| `api.provider` | string | `gemini` | API provider (`gemini` or `openai`) |
| `api.model` | string | `gemini-3-flash` | Model name for API calls |
| `api.timeout` | int | `30` | Request timeout in seconds |
| `api.max_retries` | int | `3` | Max retry attempts on failure |
| `api.retry_base` | float | `1.0` | Exponential backoff base in seconds |
| `mock` | dict | `{}` | No configuration needed; returns deterministic scores |

### `evaluators`

Per-evaluator enable/disable and custom settings.

```yaml
evaluators:
  technical_quality:
    enabled: true

  aigc_defect:
    enabled: true
    model: openai/clip-vit-large-patch14
    defect_threshold: 0.6

  product_fidelity:
    enabled: true

  compliance:
    enabled: true
    limit_words: ["最", "第一", "国家级", "顶级"]
    medical_words: ["治疗", "疗效", "药到病除"]
    banned_entities: []

  vlm_judge:
    enabled: true
    backend: mock                    # Which backend to use
    api_concurrency: 4              # Parallel API calls (api backend)
    api_max_failures: 5             # Stop after N consecutive failures
    dimensions_main_image: [sellpoint_coverage, cross_modal]
    dimensions_external: [hook_strength, marketing_logic, audience_match, cross_modal]
    dimensions_general: [cross_modal]
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `*.enabled` | bool | `true` | Enable/disable the evaluator |
| `aigc_defect.defect_threshold` | float | `0.6` | Score threshold for defect detection |
| `compliance.limit_words` | list | see example | Advertising limit words to flag |
| `compliance.medical_words` | list | see example | Medical claim words to flag |
| `vlm_judge.backend` | string | `mock` | Backend name for VLM calls |
| `vlm_judge.dimensions_*` | list | per-type | Which VLM dimensions to score per video type |

### `fusion`

Score aggregation and grading.

```yaml
fusion:
  strategy: weighted_veto

  strict_veto_dims: [compliance]

  thresholds:
    A: 0.75
    B: 0.60
    C: 0.40

  veto_thresholds:
    compliance: 0.0
    product_fidelity: 0.3
    aigc_defect: 0.3

  weights_main_image:
    technical_quality: 0.15
    aigc_defect: 0.15
    product_fidelity: 0.20
    sellpoint_coverage: 0.20
    cross_modal: 0.10

  weights_external:
    technical_quality: 0.10
    aigc_defect: 0.10
    product_fidelity: 0.10
    hook_strength: 0.20
    marketing_logic: 0.15
    audience_match: 0.10
    cross_modal: 0.05

  weights_general:
    technical_quality: 0.35
    aigc_defect: 0.35
    cross_modal: 0.30
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | string | `weighted_veto` | Fusion strategy name (from registry) |
| `strict_veto_dims` | list | `[compliance]` | Dimensions where any veto = instant REJECT |
| `thresholds.A/B/C` | float | 0.75/0.60/0.40 | Grade boundaries (score >= threshold) |
| `veto_thresholds` | dict | see example | Per-dimension score floors; below = veto |
| `weights_*` | dict | per-type | Dimension weights per video type (should sum to ~1.0) |

### `batch`

Batch processing settings.

```yaml
batch:
  mode: resident      # resident | oneshot
  chunk_size: 8       # Videos processed per chunk in resident mode
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | string | `resident` | `resident` keeps models loaded between items; `oneshot` reloads per item |
| `chunk_size` | int | `8` | Number of videos per processing chunk |

### `output`

Output formatting.

```yaml
output:
  format: json         # json | jsonl
  include_meta: true
  include_evidence: true
  pretty: true
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `format` | string | `json` | Output format for batch mode |
| `include_meta` | bool | `true` | Include ReportMeta in output |
| `include_evidence` | bool | `true` | Include evidence details in dimension results |
| `pretty` | bool | `true` | Pretty-print JSON with indentation |

## CLI Override Examples

Override any nested config key with dot notation:

```bash
# Switch VLM backend
video-eval eval --video v.mp4 --video-type general \
  --set evaluators.vlm_judge.backend=api

# Change API provider and model
video-eval eval --video v.mp4 --video-type general \
  --set backends.api.provider=openai \
  --set backends.api.model=gpt-4o

# Adjust fusion weights
video-eval eval --video v.mp4 --video-type general \
  --set fusion.thresholds.A=0.80

# Disable a specific evaluator
video-eval eval --video v.mp4 --video-type general \
  --set evaluators.compliance.enabled=false

# Force CPU device
video-eval eval --video v.mp4 --video-type general --device cpu
```

## Environment Variables

API keys are read from environment variables by the respective backends:

| Variable | Backend | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | api (gemini) | Google Gemini API key |
| `OPENAI_API_KEY` | api (openai) | OpenAI API key |

```bash
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
```

## Validation

Check your config for errors and warnings:

```bash
video-eval config check
video-eval config check --config production.yaml
video-eval config check --set evaluators.vlm_judge.backend=api
```

The checker validates:
- Required fields present
- Backend referenced by evaluator exists in registry
- Weight sums are reasonable
- Device requirements satisfiable
- Optional dependency availability matches enabled features
