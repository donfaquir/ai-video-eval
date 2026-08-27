# Architecture Overview

This document describes the internal architecture of video-eval for contributors.

## Core Modules

```
video_eval/
├── __init__.py              # Package version
├── cli.py                   # Click-based CLI (eval, extract, batch, plugins, config, device)
├── core/
│   ├── base.py              # Abstract base classes (BaseEvaluator, BaseBackend, BaseExtractor, BaseFusion)
│   ├── config.py            # ConfigLoader (YAML + defaults + CLI overrides + validation)
│   ├── device.py            # DeviceManager (auto-detect cuda/mps/cpu, memory info)
│   ├── exceptions.py        # Exception hierarchy
│   ├── pipeline.py          # Pipeline orchestration (13-step single + batch)
│   ├── registry.py          # Registry[T], Placeholder, decorators, entry-point discovery
│   └── schemas.py           # Pydantic models (EvalContext, EvalResult, FinalReport, etc.)
├── evaluators/              # Built-in evaluator plugins
├── backends/                # Built-in VLM backends
├── extractors/              # Built-in context extractors
├── fusions/                 # Built-in fusion strategies
└── prompts/                 # VLM prompt templates
```

## Registry System

The registry is the backbone of the plugin architecture. It provides:

### Generic Registry[T]

```python
class Registry(Generic[T]):
    """Thread-safe, lazy-loading plugin registry."""
```

Four global instances exist:

- `evaluator_registry` — holds `BaseEvaluator` subclasses
- `extractor_registry` — holds `BaseExtractor` subclasses
- `backend_registry` — holds `BaseBackend` subclasses
- `fusion_registry` — holds `BaseFusion` subclasses

### Registration Flow

```
1. Module imported (scan_directory or entry-point discovery)
2. @register_evaluator("name") decorator fires
3. Registry.register() stores class + extracts PluginMeta from class attributes
4. Plugin available via Registry.get("name")
```

### Lazy Loading (Placeholder)

Entry-point plugins are registered as `Placeholder` objects. The actual class is only imported (materialized) when first accessed via `registry.get(name)`. This keeps startup fast.

```
registry.register("foo", cls=Placeholder(...))  # Fast, no import
...
cls = registry.get("foo")  # Triggers loader(), imports module, validates subclass
```

Failed materializations are cached in `_broken` dict to avoid repeated import attempts.

### Discovery Order

```
initialize_registries():
  1. Set base_cls on each registry (for subclass validation)
  2. scan_directory() — import all modules under video_eval/{evaluators,backends,extractors,fusions}/
  3. discover_entry_points() — scan installed packages for entry-point declarations
```

Built-in plugins (from scan) take priority over entry-point plugins with the same name.

## Pipeline Flow

The `Pipeline.run()` method executes a 13-step flow for single-video evaluation:

```
 1. Materialize entry-point plugins
 2. Pre-flight checks (validate video_type, backend exists)
 3. Discover evaluators (filter by enabled + device)
 4. Compute available_fields (what's possible given product_info)
 5. Filter evaluators (requires satisfaction check)
 6. Run extractors (topological sort by requires/provides)
 7. Re-filter evaluators (post-extraction field availability)
 8. Run evaluators (serial, context-managed)
 9. Collect results
10. Run fusion strategy
11. Build FinalReport
12. Attach ReportMeta
13. Return report
```

### Batch Mode

`Pipeline.run_batch()` operates in **resident mode** by default:

- Extractors and evaluators are initialized once (`__enter__`)
- Multiple videos are processed sequentially through the same instances
- Resources are released at the end (`__exit__`)

This avoids repeated model loading for large batches.

## Data Model Relationships

```
ProductInfo ─┐
             ├──▶ EvalContext ──▶ ReadonlyEvalContext ──▶ Evaluator.evaluate()
VideoMeta ───┤                                                    │
Frames ──────┤                                                    ▼
ASR ─────────┤                                              EvalResult (per dimension)
OCR ─────────┤                                                    │
CLIP ────────┘                                                    ▼
                                                            Fusion.fuse()
                                                                  │
                                                                  ▼
                                                          FusionOutcome
                                                                  │
                                                                  ▼
                                                           FinalReport
```

### Key Models (in `video_eval/core/schemas.py`)

| Model | Purpose |
|-------|---------|
| `EvalContext` | Mutable accumulator; extractors write fields into it |
| `ReadonlyEvalContext` | Frozen view passed to evaluators (Pydantic `frozen=True`) |
| `EvalResult` | Single dimension score with evidence and metadata |
| `FusionOutcome` | Aggregated score, grade, pass/fail, veto reasons |
| `FinalReport` | Complete output including all dimensions + meta |
| `PluginMeta` | Registry metadata extracted from class attributes |
| `BatchItem` / `BatchItemResult` | Batch processing I/O |

## Plugin Lifecycle

### Evaluator Lifecycle

```
1. Class registered (module import time)
2. Pipeline discovers evaluator (list_meta, filter by device/requires)
3. Pipeline instantiates: cls(device_manager, config)
4. Pipeline calls __enter__() — load models, allocate GPU
5. Pipeline calls evaluate(context) — one or more times (batch)
6. Pipeline calls __exit__() — release resources
```

### Extractor Lifecycle

Same as evaluator, but with `extract(context)` instead of `evaluate(context)`. Extractors run in topological order based on `requires`/`provides` dependencies.

### Backend Lifecycle

Backends are managed by the evaluator that uses them (typically `vlm_judge`). The evaluator's `__enter__` initializes the backend; its `__exit__` tears it down.

## DeviceManager

```python
class DeviceManager:
    """Detects best available device and provides memory info."""

    device_type: str   # "cuda" | "mps" | "cpu"

    def satisfies(self, requirement: str) -> bool:
        """Check if current device meets a plugin's requirement."""

    def memory_info(self) -> dict:
        """Return total_gb, free_gb, used_gb."""

    def is_gpu(self) -> bool:
        """True if device is cuda or mps."""
```

Device requirement resolution:
- `"any"` — always satisfied
- `"cpu"` — always satisfied
- `"gpu"` — satisfied by cuda or mps
- `"cuda"` — only satisfied by cuda
- `"mps"` — only satisfied by mps

## ConfigLoader

```python
class ConfigLoader:
    def load(self) -> dict: ...
    def merge_cli_overrides(self, config: dict, overrides: list[str]) -> dict: ...
    def validate(self, config: dict) -> list[ValidationIssue]: ...
```

Validation checks:
- Backend referenced by evaluator exists
- Weight keys match active dimensions
- Required fields present
- Type correctness

## Error Handling

Exception hierarchy (in `video_eval/core/exceptions.py`):

```
VideoEvalError (base)
├── ConfigError           — Invalid configuration
├── ExtractionError       — Required extractor failed
├── RegistryError         — Generic registry issue
│   ├── DuplicateRegistrationError
│   ├── NameNotFoundError
│   ├── RegistryFrozenError
│   └── MaterializationError
└── VLMOutputParseError   — Backend output unparseable
```

The CLI maps these to exit codes (see `cli.py:_compute_exit_code`).

## Adding a New Built-in Plugin

1. Create `video_eval/{type}/{name}.py`
2. Subclass the appropriate base class
3. Apply the registration decorator (`@register_evaluator`, `@register_backend`, etc.)
4. The module is auto-discovered by `scan_directory()` at startup
5. Add tests in `tests/test_{name}.py`
6. Update `config.yaml.example` with default settings
