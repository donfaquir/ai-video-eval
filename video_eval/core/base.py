"""Abstract base classes for all plugin types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from video_eval.core.device import DeviceManager

if TYPE_CHECKING:
    from video_eval.core.schemas import EvalResult, FusionOutcome, ReadonlyEvalContext, VLMResult


class BaseEvaluator(ABC):
    """Abstract base class for evaluators.

    Evaluators score one or more quality dimensions of a video.
    They are context-managed: __enter__ loads heavy resources,
    __exit__ releases them.
    """

    # --- Class attributes (plugin metadata) ---
    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    device_requirement: ClassVar[str] = "any"
    requires: ClassVar[list[str]] = []
    config_schema: ClassVar[dict] = {}
    backend_config_key: ClassVar[str | None] = None
    dimension_slots: ClassVar[dict[str, list[str]] | None] = None
    default_weights: ClassVar[dict[str, float] | float | None] = None

    def __init__(self, device_manager: DeviceManager, config: dict) -> None:
        """Store references only. NO heavy resource loading."""
        self.device_manager = device_manager
        self.config = config

    @abstractmethod
    def __enter__(self) -> BaseEvaluator:
        """Load models / allocate GPU memory. Must try/except self-cleanup on failure."""
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release resources. Must be idempotent."""
        ...

    @abstractmethod
    def evaluate(self, context: ReadonlyEvalContext) -> EvalResult | list[EvalResult]:
        """Core evaluation. Returns single or multiple results."""
        ...

    def check_availability(self) -> tuple[bool, str]:
        """Check runtime dynamic conditions (after __enter__).

        Returns:
            Tuple of (available, human_readable_reason).
            Reason text goes to verbose log only;
            EvalResult.reason = "runtime_unavailable".
        """
        return True, ""

    def slots_for(self, video_type: str) -> list[str]:
        """Return result slot names for given video_type.

        Single-slot evaluators always return [self.name].
        Multi-slot evaluators check config overrides first,
        then fall back to dimension_slots mapping.
        """
        if self.dimension_slots is None:
            return [self.name]
        override_key = f"dimensions_{video_type}"
        override = self.config.get(override_key)
        if override is not None:  # explicit is-not-None check; [] is valid "disable"
            return override
        if video_type not in self.dimension_slots:
            raise ValueError(f"Unknown video_type '{video_type}' for evaluator '{self.name}'")
        return self.dimension_slots[video_type]


class BaseBackend(ABC):
    """Abstract base class for VLM backends.

    Backends wrap a vision-language model and expose a single
    judge() call that returns structured scoring output.
    """

    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    device_requirement: ClassVar[str] = "any"
    requires: ClassVar[list[str]] = []
    config_schema: ClassVar[dict] = {}

    def __init__(self, device_manager: DeviceManager, config: dict) -> None:
        """Initialize with device manager and backend config section."""
        self.device_manager = device_manager
        self.config = config

    @abstractmethod
    def __enter__(self) -> BaseBackend:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release model resources. Must be idempotent."""
        ...

    @abstractmethod
    def judge(self, context: ReadonlyEvalContext, prompt: str) -> VLMResult:
        """Single judge call.

        Context is read-only (DO NOT mutate).
        Must parse raw output internally; raise VLMOutputParseError on failure.
        """
        ...


class BaseExtractor(ABC):
    """Abstract base class for context extractors.

    Extractors produce derived fields (e.g., keyframes, embeddings)
    that evaluators consume via the shared EvalContext.
    """

    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    provides: ClassVar[list[str]]
    requires: ClassVar[list[str]] = []
    criticality: ClassVar[str] = "optional"
    device_requirement: ClassVar[str] = "any"
    config_schema: ClassVar[dict] = {}

    def __init__(self, device_manager: DeviceManager, config: dict) -> None:
        """Store references only. NO heavy resource loading."""
        self.device_manager = device_manager
        self.config = config

    @abstractmethod
    def __enter__(self) -> BaseExtractor:
        """Load models / allocate resources."""
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release resources. Must be idempotent."""
        ...

    @abstractmethod
    def extract(self, context: ReadonlyEvalContext) -> dict:
        """Execute extraction. Returns incremental field dict.

        Keys MUST be a subset of self.provides.
        Context is read-only; may consume prior extractor outputs.
        """
        ...


class BaseFusion(ABC):
    """Abstract base class for score fusion strategies.

    Fusion takes per-dimension EvalResults and produces a single
    FusionOutcome with an aggregated score. Intentionally does NOT
    receive a DeviceManager (pure computation, no GPU needed).
    """

    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    requires: ClassVar[list[str]] = []
    config_schema: ClassVar[dict] = {}

    def __init__(self, config: dict) -> None:
        """Receives fusion config section.

        NOTE: intentionally does NOT take device_manager (pure computation).
        """
        self.config = config

    @abstractmethod
    def fuse(
        self,
        results: dict[str, EvalResult],
        video_type: str,
        weights: dict,
        default_weights: dict[str, float | None],
    ) -> FusionOutcome:
        """Aggregate dimension scores into final outcome.

        Args:
            results: dimension_name -> EvalResult (includes placeholders).
            video_type: The type of video being evaluated.
            weights: From config fusion.weights_<video_type>.
            default_weights: Pre-computed by Pipeline from evaluator metas.
        """
        ...
