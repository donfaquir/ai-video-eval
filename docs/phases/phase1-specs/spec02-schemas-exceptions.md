# Spec 02：数据模型与异常定义

## 目标

实现所有 pydantic 数据模型（`schemas.py`）和框架级异常（`exceptions.py`）。这些是纯数据定义，无外部副作用，是后续所有 spec 的基础类型。

## 依赖

Spec 01（项目结构存在）。

## 产出文件

- `video_eval/core/schemas.py`
- `video_eval/core/exceptions.py`

## schemas.py 完整定义

参考详设 §3 全节。所有模型继承 `pydantic.BaseModel`，`model_config = ConfigDict(arbitrary_types_allowed=True)`。

### EvalContext

```python
from pydantic import BaseModel, ConfigDict
from typing import Any

class VideoMeta(BaseModel):
    resolution: tuple[int, int]
    duration: float
    fps: float
    bitrate: int
    has_audio: bool

class FrameItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame_idx: int
    timestamp: float
    image: Any  # PIL Image or ndarray

class AsrResult(BaseModel):
    full_text: str
    segments: list[dict]  # each has start/end/text
    language: str

class OcrItem(BaseModel):
    frame_idx: int
    timestamp: float
    text: str
    bbox: list[float]

class ProductInfo(BaseModel):
    title: str
    selling_points: list[str]
    main_image_paths: list[str]

class EvalContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_path: str
    video_meta: VideoMeta | None = None
    frames: list[FrameItem] | None = None
    asr: AsrResult | None = None
    ocr: list[OcrItem] | None = None
    clip_features: Any | None = None
    product_info: ProductInfo | None = None
    video_type: str  # "main_image" / "external" / "general"
    extraction_failures: dict[str, str] = {}  # field -> failure reason

    def readonly(self) -> "ReadonlyEvalContext":
        """Return a frozen view sharing field objects (no deep copy)."""
        return ReadonlyEvalContext(**self.model_dump())

    def merge(self, feats: dict, declared_provides: list[str]) -> None:
        """Merge extractor output into context with bidirectional validation."""
        for key in feats:
            if key not in declared_provides:
                raise ValueError(f"Key '{key}' not in declared provides {declared_provides}")
            if not hasattr(self, key):
                raise ValueError(f"Key '{key}' is not a known EvalContext field")
        for key, value in feats.items():
            setattr(self, key, value)

class ReadonlyEvalContext(BaseModel):
    """Frozen view of EvalContext. Assignment raises ValidationError."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    video_path: str
    video_meta: VideoMeta | None = None
    frames: list[FrameItem] | None = None
    asr: AsrResult | None = None
    ocr: list[OcrItem] | None = None
    clip_features: Any | None = None
    product_info: ProductInfo | None = None
    video_type: str
    extraction_failures: dict[str, str] = {}
```

### EvalResult

```python
class EvalResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: str        # slot name
    evaluator: str        # evaluator name that produced this
    score: float          # [0.0, 1.0]; 0.0 when skipped/error
    status: str           # "scored" / "skipped" / "error"
    reason: str | None = None  # machine-readable skip/error reason
    evidence: Any = None
    reasoning: str | None = None
    suggestion: str | None = None
```

### VLMResult

```python
class EvidenceItem(BaseModel):
    modality: str         # "visual" / "audio" / "text"
    timestamp: float | None = None
    detail: str

class VLMResult(BaseModel):
    score: float
    reasoning: str
    evidence: list[EvidenceItem]
    suggestion: str
    raw_output: str
```

### FinalReport & ReportMeta

```python
class ReportMeta(BaseModel):
    framework_version: str
    device: str
    backend: str
    vlm_model: str
    evaluator_versions: dict[str, str]
    skipped: list[str]
    config_hash: str
    timestamp: str  # ISO-8601

class FinalReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_path: str
    video_type: str
    overall_score: float
    grade: str           # "A" / "B" / "C" / "REJECT"
    passed: bool         # A/B = True
    veto_reasons: list[str]
    dimension_results: dict[str, EvalResult]
    suggestions: list[str]
    meta: ReportMeta
```

### FusionOutcome

```python
class FusionOutcome(BaseModel):
    overall_score: float
    grade: str
    passed: bool
    veto_reasons: list[str]
    suggestions: list[str]
```

### BatchItem & BatchItemResult

```python
class BatchItem(BaseModel):
    video_path: str
    video_type: str
    product_info: ProductInfo | None = None

class BatchItemResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    item: BatchItem
    report: FinalReport | None = None
    error: str | None = None
```

### ValidationIssue

```python
class ValidationIssue(BaseModel):
    severity: str         # "error" / "warning"
    plugin_name: str
    message: str
    field: str | None = None
```

### PluginMeta

```python
class PluginMeta(BaseModel):
    name: str
    version: str
    device_requirement: str       # "cuda" / "mps" / "gpu" / "any"
    requires: list[str]
    config_schema: dict
    provides: list[str]           # extractors only
    criticality: str              # extractors only: "required" / "optional"
    backend_config_key: str | None  # evaluators only
    default_weights: dict[str, float] | float | None
    dimension_slots: dict[str, list[str]] | None
    origin: str                   # "builtin" / "entry-point" / "explicit"
    is_placeholder: bool
```

### EvaluatorInfo

```python
class EvaluatorInfo(BaseModel):
    meta: PluginMeta
    config: dict
    effective_slots: list[str]
    status: str = "pending"       # "pending" / "active" / "skipped" / "error"
    reason: str | None = None
```

## exceptions.py 完整定义

```python
class VideoEvalError(Exception):
    """Base for all video-eval exceptions."""
    pass

# --- Registry exceptions ---

class RegistryError(VideoEvalError):
    """Base for Registry errors."""
    def __init__(self, message: str, registry_name: str = ""):
        self.registry_name = registry_name
        super().__init__(message)

class NameNotFoundError(RegistryError):
    """get() miss. Message contains suggestions."""
    pass

class RegistryFrozenError(RegistryError):
    """Attempted registration after freeze."""
    pass

class DuplicateRegistrationError(RegistryError):
    """Concrete→Concrete conflict without allow_override."""
    pass

class MaterializationError(RegistryError):
    """Lazy load (import or type check) failed."""
    pass

# --- Pipeline exceptions ---

class ExtractionError(VideoEvalError):
    """Required extractor failed. Carries extractor name."""
    def __init__(self, extractor_name: str):
        self.extractor_name = extractor_name
        super().__init__(f"Required extractor '{extractor_name}' failed")

class ConfigError(VideoEvalError):
    """Configuration validation failure (exit code 2)."""
    pass

# --- Backend exceptions ---

class VLMOutputParseError(VideoEvalError):
    """Backend failed to parse VLM output."""
    def __init__(self, message: str, raw_output: str = ""):
        self.raw_output = raw_output[:500]  # truncate for safety
        super().__init__(message)
```

## 验收标准

- [ ] `from video_eval.core.schemas import EvalContext, EvalResult, FinalReport, PluginMeta` 成功
- [ ] `from video_eval.core.exceptions import RegistryError, ExtractionError, ConfigError` 成功
- [ ] `EvalContext(video_path="/tmp/a.mp4", video_type="general")` 构造成功
- [ ] `EvalContext(...).readonly()` 返回冻结视图，赋值抛 ValidationError
- [ ] `EvalContext.merge({"asr": ...}, declared_provides=["asr"])` 正常工作
- [ ] `EvalContext.merge({"unknown": ...}, declared_provides=["unknown"])` 抛 ValueError
- [ ] 所有异常可正常 raise/catch/str()
