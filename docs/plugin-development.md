# Plugin Development Guide

## Architecture Overview

video-eval uses a registry-based plugin system with four extension points:

| Type | Base Class | Registry | Entry-Point Group |
|------|-----------|----------|-------------------|
| Evaluator | `BaseEvaluator` | `evaluator_registry` | `video_eval.evaluators` |
| Backend | `BaseBackend` | `backend_registry` | `video_eval.backends` |
| Extractor | `BaseExtractor` | `extractor_registry` | `video_eval.extractors` |
| Fusion | `BaseFusion` | `fusion_registry` | `video_eval.fusions` |

Plugins are discovered in two ways:
1. **Built-in scan** — All modules under `video_eval/evaluators/`, `video_eval/backends/`, etc. are auto-imported at startup
2. **Entry-points** — Third-party packages declare plugins in their `pyproject.toml`; discovered via `importlib.metadata.entry_points()`

## Creating a Custom Evaluator

### Full Example

```python
"""video_eval_watermark/evaluator.py — Watermark detection evaluator."""

from video_eval.core.base import BaseEvaluator
from video_eval.core.registry import register_evaluator
from video_eval.core.schemas import EvalResult, ReadonlyEvalContext


@register_evaluator("watermark_detect")
class WatermarkEvaluator(BaseEvaluator):
    """Detects visible watermarks in video frames."""

    # --- Plugin metadata (class attributes) ---
    name = "watermark_detect"
    version = "1.0.0"
    device_requirement = "any"       # "any" | "cuda" | "mps" | "gpu"
    requires = ["frames"]            # EvalContext fields this evaluator needs
    config_schema = {
        "threshold": {"type": "float", "default": 0.5},
        "check_corners": {"type": "bool", "default": True},
    }

    def __enter__(self):
        """Load detection model or resources."""
        self.threshold = self.config.get("threshold", 0.5)
        self.check_corners = self.config.get("check_corners", True)
        # Load your model here if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resources. Must be idempotent."""
        pass

    def evaluate(self, context: ReadonlyEvalContext) -> EvalResult:
        """Score the video on watermark presence."""
        if context.frames is None:
            return EvalResult(
                dimension="watermark_detect",
                evaluator=self.name,
                score=0.0,
                status="skipped",
                reason="missing_field:frames",
            )

        # Your detection logic here
        watermark_count = 0
        for frame in context.frames:
            if self._has_watermark(frame.image):
                watermark_count += 1

        ratio = watermark_count / len(context.frames) if context.frames else 0
        score = 1.0 - ratio  # Higher score = fewer watermarks

        return EvalResult(
            dimension="watermark_detect",
            evaluator=self.name,
            score=score,
            status="scored",
            evidence={"watermark_frame_ratio": ratio},
            reasoning=f"Detected watermarks in {watermark_count}/{len(context.frames)} frames",
            suggestion="Remove watermarks before publishing" if score < 0.8 else None,
        )

    def _has_watermark(self, image) -> bool:
        """Placeholder detection logic."""
        # Implement your watermark detection here
        return False
```

### Key Rules

1. **`name` class attribute must match the decorator argument** — `@register_evaluator("foo")` requires `name = "foo"`
2. **`__enter__` does all heavy loading** — Constructor (`__init__`) only stores references
3. **`__exit__` must be idempotent** — May be called multiple times or after partial `__enter__`
4. **`evaluate` receives `ReadonlyEvalContext`** — Do not mutate the context
5. **Return `EvalResult` with proper status** — `"scored"`, `"skipped"`, or `"error"`

### Multi-Dimension Evaluator

For evaluators that produce multiple dimension scores (like `vlm_judge`):

```python
@register_evaluator("multi_scorer")
class MultiScorer(BaseEvaluator):
    name = "multi_scorer"
    requires = ["frames", "asr"]
    # Map video_type -> list of dimension slots
    dimension_slots = {
        "main_image": ["visual_appeal", "brand_consistency"],
        "external": ["visual_appeal"],
        "general": ["visual_appeal"],
    }

    def evaluate(self, context: ReadonlyEvalContext) -> list[EvalResult]:
        """Return one EvalResult per active slot."""
        slots = self.slots_for(context.video_type)
        results = []
        for slot in slots:
            score = self._score_dimension(slot, context)
            results.append(EvalResult(
                dimension=slot,
                evaluator=self.name,
                score=score,
                status="scored",
            ))
        return results
```

## Creating a Custom Backend

### Full Example

```python
"""video_eval_anthropic/backend.py — Anthropic Claude backend."""

import json
from video_eval.core.base import BaseBackend
from video_eval.core.registry import register_backend
from video_eval.core.schemas import EvidenceItem, ReadonlyEvalContext, VLMResult


@register_backend("anthropic")
class AnthropicBackend(BaseBackend):
    """VLM backend using Anthropic Claude API."""

    name = "anthropic"
    version = "1.0.0"
    device_requirement = "any"
    config_schema = {
        "model": {"type": "str", "default": "claude-sonnet-4-20250514"},
        "timeout": {"type": "int", "default": 60},
        "max_tokens": {"type": "int", "default": 4096},
    }

    def __enter__(self):
        """Initialize API client."""
        import anthropic
        self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        self.model = self.config.get("model", "claude-sonnet-4-20250514")
        self.timeout = self.config.get("timeout", 60)
        self.max_tokens = self.config.get("max_tokens", 4096)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up client."""
        self.client = None

    def judge(self, context: ReadonlyEvalContext, prompt: str) -> VLMResult:
        """Send frames + prompt to Claude and parse structured response."""
        # Build message with images
        content = self._build_content(context, prompt)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )

        raw_output = response.content[0].text
        return self._parse_output(raw_output)

    def _build_content(self, context, prompt):
        """Build multimodal message content."""
        # Implementation: encode frames as base64, build content blocks
        ...

    def _parse_output(self, raw: str) -> VLMResult:
        """Parse structured JSON from model output."""
        data = json.loads(raw)
        return VLMResult(
            score=data["score"],
            reasoning=data["reasoning"],
            evidence=[EvidenceItem(**e) for e in data.get("evidence", [])],
            suggestion=data.get("suggestion", ""),
            raw_output=raw,
        )
```

### Backend Contract

- `judge()` receives a `ReadonlyEvalContext` and a prompt string
- Must return a `VLMResult` with parsed score, reasoning, evidence, and suggestion
- Raise `VLMOutputParseError` (from `video_eval.core.exceptions`) if the model output cannot be parsed

## Creating a Custom Extractor

```python
"""video_eval_scene/extractor.py — Scene classification extractor."""

from video_eval.core.base import BaseExtractor
from video_eval.core.registry import register_extractor
from video_eval.core.schemas import ReadonlyEvalContext


@register_extractor("scene_classify")
class SceneClassifier(BaseExtractor):
    """Classifies video scenes (indoor/outdoor/studio/etc)."""

    name = "scene_classify"
    version = "1.0.0"
    provides = ["scene_labels"]      # Fields this extractor adds to EvalContext
    requires = ["frames"]            # Must run after frame extraction
    criticality = "optional"         # "required" | "optional"
    device_requirement = "any"

    def __enter__(self):
        """Load classification model."""
        # self.model = load_scene_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release model."""
        pass

    def extract(self, context: ReadonlyEvalContext) -> dict:
        """Classify scenes in extracted frames.

        Returns dict with keys that are a subset of self.provides.
        """
        if context.frames is None:
            return {"scene_labels": []}

        labels = []
        for frame in context.frames:
            label = self._classify(frame.image)
            labels.append({"frame_idx": frame.frame_idx, "label": label})

        return {"scene_labels": labels}

    def _classify(self, image) -> str:
        return "indoor"  # Placeholder
```

**Important**: To use custom `provides` fields, you must also add the corresponding field to `EvalContext` in the core schema, or use a dynamic approach.

## Publishing as a pip Package

### 1. Package structure

```
video-eval-watermark/
├── pyproject.toml
├── src/
│   └── video_eval_watermark/
│       ├── __init__.py
│       └── evaluator.py
```

### 2. Declare entry-points in `pyproject.toml`

```toml
[project]
name = "video-eval-watermark"
version = "1.0.0"
dependencies = ["video-eval>=0.1.0"]

[project.entry-points."video_eval.evaluators"]
watermark_detect = "video_eval_watermark.evaluator:WatermarkEvaluator"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3. Install and verify

```bash
pip install -e ./video-eval-watermark
video-eval plugins --detail watermark_detect
```

The plugin will appear with `origin: entry-point` in the plugins list.

### Entry-point groups

| Plugin Type | Group |
|-------------|-------|
| Evaluator | `video_eval.evaluators` |
| Backend | `video_eval.backends` |
| Extractor | `video_eval.extractors` |
| Fusion | `video_eval.fusions` |

## Plugin Development Checklist

- [ ] Class attribute `name` matches decorator registration name
- [ ] `device_requirement` accurately reflects hardware needs
- [ ] `requires` lists all EvalContext fields the plugin reads
- [ ] `provides` (extractors only) lists all fields the extractor writes
- [ ] `__enter__` handles import errors gracefully (wrap optional deps in try/except)
- [ ] `__exit__` is idempotent and releases all resources
- [ ] `config_schema` documents all accepted config keys
- [ ] Plugin works with `video-eval plugins --detail <name>` (metadata is correct)
- [ ] Tests cover both success and degraded (missing field) paths
- [ ] Entry-point declared in `pyproject.toml` if distributing as a package

## Testing Your Plugin

```python
import pytest
from video_eval.core.device import DeviceManager
from video_eval.core.schemas import EvalContext, FrameItem

from video_eval_watermark.evaluator import WatermarkEvaluator


def test_watermark_evaluator():
    dm = DeviceManager()
    config = {"threshold": 0.5}

    evaluator = WatermarkEvaluator(dm, config)
    with evaluator:
        context = EvalContext(
            video_path="test.mp4",
            video_type="general",
            frames=[FrameItem(frame_idx=0, timestamp=0.0, image=mock_image())],
        )
        result = evaluator.evaluate(context.readonly())

    assert result.status == "scored"
    assert 0.0 <= result.score <= 1.0
    assert result.dimension == "watermark_detect"


def test_watermark_evaluator_no_frames():
    dm = DeviceManager()
    evaluator = WatermarkEvaluator(dm, {})
    with evaluator:
        context = EvalContext(video_path="test.mp4", video_type="general")
        result = evaluator.evaluate(context.readonly())

    assert result.status == "skipped"
    assert result.reason == "missing_field:frames"
```
