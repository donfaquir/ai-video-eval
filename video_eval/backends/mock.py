"""Mock backend for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from video_eval.core.base import BaseBackend
from video_eval.core.registry import register_backend
from video_eval.core.schemas import VLMResult

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext


@register_backend("mock")
class MockBackend(BaseBackend):
    """Mock VLM backend that returns fixed scores for testing."""

    name = "mock"
    version = "0.1.0"
    device_requirement = "any"
    config_schema: dict = {}

    def __init__(self, device_manager, config: dict) -> None:  # noqa: ANN001
        """Store default_score from config (default 0.75)."""
        super().__init__(device_manager, config)
        self._default_score = config.get("default_score", 0.75)

    def __enter__(self) -> MockBackend:
        """No-op: no model to load."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """No-op: no resources to release."""

    def judge(self, context: ReadonlyEvalContext, prompt: str) -> VLMResult:
        """Return a fixed-score VLMResult for testing purposes."""
        return VLMResult(
            score=self._default_score,
            reasoning="Mock evaluation - fixed score",
            evidence=[],
            suggestion="",
            raw_output=f"mock: score={self._default_score}",
        )
