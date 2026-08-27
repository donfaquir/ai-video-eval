# Spec 05：基类定义

## 目标

实现 `BaseEvaluator`、`BaseBackend`、`BaseExtractor`、`BaseFusion` 四个抽象基类。定义插件契约，供后续所有内置和第三方插件继承。

## 依赖

Spec 02（schemas）、Spec 03（DeviceManager）。

## 产出文件

- `video_eval/core/base.py`

## BaseEvaluator

参考详设 §2.1。

```python
from abc import ABC, abstractmethod
from typing import ClassVar

class BaseEvaluator(ABC):
    # --- Class attributes (plugin metadata) ---
    name: ClassVar[str]                                    # required
    version: ClassVar[str] = "0.1.0"
    device_requirement: ClassVar[str] = "any"              # "cuda"/"mps"/"gpu"/"any"
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
    def __enter__(self) -> "BaseEvaluator":
        """Load models / allocate GPU memory. Must try/except self-cleanup on failure."""
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release resources. Must be idempotent."""
        ...

    @abstractmethod
    def evaluate(self, context: "ReadonlyEvalContext") -> "EvalResult | list[EvalResult]":
        """Core evaluation. Returns single or multiple results."""
        ...

    def check_availability(self) -> tuple[bool, str]:
        """
        Check runtime dynamic conditions (after __enter__).
        Returns (available, human_readable_reason).
        Reason text goes to verbose log only; EvalResult.reason = "runtime_unavailable".
        """
        return True, ""

    def slots_for(self, video_type: str) -> list[str]:
        """
        Return result slot names for given video_type.
        - Single-slot: [self.name]
        - Multi-slot: config override (dimensions_<video_type>) if not None, else dimension_slots[video_type]
        """
        if self.dimension_slots is None:
            return [self.name]
        override_key = f"dimensions_{video_type}"
        override = self.config.get(override_key)
        if override is not None:
            return override
        if video_type not in self.dimension_slots:
            raise ValueError(f"Unknown video_type '{video_type}' for evaluator '{self.name}'")
        return self.dimension_slots[video_type]
```

## BaseBackend

参考详设 §2.2。

```python
class BaseBackend(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    device_requirement: ClassVar[str] = "any"
    requires: ClassVar[list[str]] = []
    config_schema: ClassVar[dict] = {}

    def __init__(self, device_manager: DeviceManager, config: dict) -> None:
        """config = config["backends"][self.name] section."""
        self.device_manager = device_manager
        self.config = config

    @abstractmethod
    def __enter__(self) -> "BaseBackend": ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    @abstractmethod
    def judge(self, context: "ReadonlyEvalContext", prompt: str) -> "VLMResult":
        """
        Single judge call. context is read-only (DO NOT mutate).
        Must parse raw output internally; raise VLMOutputParseError on failure.
        """
        ...
```

## BaseExtractor

参考详设 §2.3。

```python
class BaseExtractor(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    provides: ClassVar[list[str]]               # required: produced context fields
    requires: ClassVar[list[str]] = []          # consumed context fields (topo sort)
    criticality: ClassVar[str] = "optional"     # "required" / "optional"
    device_requirement: ClassVar[str] = "any"
    config_schema: ClassVar[dict] = {}

    def __init__(self, device_manager: DeviceManager, config: dict) -> None:
        self.device_manager = device_manager
        self.config = config

    @abstractmethod
    def __enter__(self) -> "BaseExtractor": ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    @abstractmethod
    def extract(self, context: "ReadonlyEvalContext") -> dict:
        """
        Execute extraction. Returns incremental field dict.
        Keys MUST be subset of self.provides.
        Context is read-only; may consume prior extractor outputs (e.g., frames).
        """
        ...
```

## BaseFusion

参考详设 §2.4。

```python
class BaseFusion(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    requires: ClassVar[list[str]] = []
    config_schema: ClassVar[dict] = {}

    def __init__(self, config: dict) -> None:
        """
        Receives fusion config section.
        NOTE: intentionally does NOT take device_manager (pure computation).
        """
        self.config = config

    @abstractmethod
    def fuse(
        self,
        results: dict[str, "EvalResult"],
        video_type: str,
        weights: dict,
        default_weights: dict[str, float | None],
    ) -> "FusionOutcome":
        """
        Aggregate dimension scores into final outcome.
        - results: dimension_name -> EvalResult (includes placeholders)
        - weights: from config fusion.weights_<video_type>
        - default_weights: pre-computed by Pipeline from evaluator metas
        """
        ...
```

## 验收标准

- [ ] 所有基类可被继承并实例化（非抽象子类）
- [ ] `BaseEvaluator` 子类不实现 `evaluate` → TypeError on instantiation
- [ ] `slots_for("main_image")` 对单槽位评估器返回 `[name]`
- [ ] `slots_for("external")` 对多槽位评估器返回 dimension_slots 默认值
- [ ] `slots_for("external")` 在 config 有 `dimensions_external: [x]` 时返回 `[x]`
- [ ] `slots_for("external")` 在 config 有 `dimensions_external: []` 时返回 `[]`（空=主动关闭）
- [ ] `BaseFusion.__init__` 只接收 config（不接收 device_manager）
