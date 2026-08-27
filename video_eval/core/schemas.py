"""Pydantic data models for the video-eval framework."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class VideoMeta(BaseModel):
    """Technical metadata extracted from a video file."""

    resolution: tuple[int, int]
    duration: float
    fps: float
    bitrate: int
    has_audio: bool


class FrameItem(BaseModel):
    """A single extracted frame with its index and timestamp."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_idx: int
    timestamp: float
    image: Any  # PIL Image or ndarray


class AsrResult(BaseModel):
    """Automatic speech recognition result."""

    full_text: str
    segments: list[dict]  # each has start/end/text
    language: str


class OcrItem(BaseModel):
    """OCR detection for a single frame."""

    frame_idx: int
    timestamp: float
    text: str
    bbox: list[float]


class ProductInfo(BaseModel):
    """Product information for main-image video evaluation."""

    title: str
    selling_points: list[str]
    main_image_paths: list[str]


class EvalContext(BaseModel):
    """Mutable evaluation context that accumulates extracted features."""

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

    def readonly(self) -> ReadonlyEvalContext:
        """Return a frozen view sharing field objects (no deep copy)."""
        return ReadonlyEvalContext(**self.model_dump())

    def merge(self, feats: dict, declared_provides: list[str]) -> None:
        """Merge extractor output into context with bidirectional validation."""
        for key in feats:
            if key not in declared_provides:
                raise ValueError(
                    f"Key '{key}' not in declared provides {declared_provides}"
                )
            if not hasattr(self, key):
                raise ValueError(
                    f"Key '{key}' is not a known EvalContext field"
                )
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


class EvalResult(BaseModel):
    """Result from a single evaluator for one dimension."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: str  # slot name
    evaluator: str  # evaluator name that produced this
    score: float  # [0.0, 1.0]; 0.0 when skipped/error
    status: str  # "scored" / "skipped" / "error"
    reason: str | None = None  # machine-readable skip/error reason
    evidence: Any = None
    reasoning: str | None = None
    suggestion: str | None = None


class EvidenceItem(BaseModel):
    """A single piece of evidence from VLM evaluation."""

    modality: str  # "visual" / "audio" / "text"
    timestamp: float | None = None
    detail: str


class VLMResult(BaseModel):
    """Parsed output from a VLM backend call."""

    score: float
    reasoning: str
    evidence: list[EvidenceItem]
    suggestion: str
    raw_output: str


class ReportMeta(BaseModel):
    """Metadata attached to a final evaluation report."""

    framework_version: str
    device: str
    backend: str
    vlm_model: str
    evaluator_versions: dict[str, str]
    skipped: list[str]
    config_hash: str
    timestamp: str  # ISO-8601


class FinalReport(BaseModel):
    """Complete evaluation report for a single video."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_path: str
    video_type: str
    overall_score: float
    grade: str  # "A" / "B" / "C" / "REJECT"
    passed: bool  # A/B = True
    veto_reasons: list[str]
    dimension_results: dict[str, EvalResult]
    suggestions: list[str]
    meta: ReportMeta


class FusionOutcome(BaseModel):
    """Output from the fusion strategy (score aggregation + grading)."""

    overall_score: float
    grade: str
    passed: bool
    veto_reasons: list[str]
    suggestions: list[str]


class BatchItem(BaseModel):
    """A single item in a batch evaluation request."""

    video_path: str
    video_type: str
    product_info: ProductInfo | None = None


class BatchItemResult(BaseModel):
    """Result of evaluating a single batch item."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: BatchItem
    report: FinalReport | None = None
    error: str | None = None


class ValidationIssue(BaseModel):
    """An issue found during plugin or config validation."""

    severity: str  # "error" / "warning"
    plugin_name: str
    message: str
    field: str | None = None


class PluginMeta(BaseModel):
    """Metadata describing a registered plugin (extractor or evaluator)."""

    name: str
    version: str
    device_requirement: str  # "cuda" / "mps" / "gpu" / "any"
    requires: list[str]
    config_schema: dict
    provides: list[str]  # extractors only
    criticality: str  # extractors only: "required" / "optional"
    backend_config_key: str | None  # evaluators only
    default_weights: dict[str, float] | float | None
    dimension_slots: dict[str, list[str]] | None
    origin: str  # "builtin" / "entry-point" / "explicit"
    is_placeholder: bool


class EvaluatorInfo(BaseModel):
    """Runtime info for a resolved evaluator in the pipeline."""

    meta: PluginMeta
    config: dict
    effective_slots: list[str]
    status: str = "pending"  # "pending" / "active" / "skipped" / "error"
    reason: str | None = None
