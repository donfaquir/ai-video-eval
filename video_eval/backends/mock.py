"""Mock backend for testing."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from video_eval.core.base import BaseBackend
from video_eval.core.registry import register_backend
from video_eval.core.schemas import EvidenceItem, VLMResult

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext


@register_backend("mock")
class MockBackend(BaseBackend):
    """Mock VLM backend that returns configurable scores for testing."""

    name = "mock"
    version = "0.2.0"
    device_requirement = "any"
    config_schema: dict = {
        "default_score": {"type": "float", "default": 0.75},
        "dimension_scores": {"type": "dict", "default": {}},
        "delay": {"type": "float", "default": 0.0},
    }

    # Known dimension names for extraction from prompt text
    _KNOWN_DIMENSIONS = (
        "sellpoint_coverage",
        "cross_modal",
        "hook_strength",
        "marketing_logic",
        "audience_match",
    )

    def __init__(self, device_manager, config: dict) -> None:  # noqa: ANN001
        """Store config: default_score, dimension_scores, delay."""
        super().__init__(device_manager, config)
        self._default_score = config.get("default_score", 0.75)
        self._dimension_scores: dict[str, float] = config.get("dimension_scores", {})
        self._delay: float = config.get("delay", 0.0)

    def __enter__(self) -> MockBackend:
        """No-op: no model to load."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """No-op: no resources to release."""

    def judge(self, context: ReadonlyEvalContext, prompt: str) -> VLMResult:
        """Return a configurable VLMResult based on dimension extracted from prompt."""
        # Simulate delay for timeout testing
        if self._delay > 0:
            time.sleep(self._delay)

        # Extract dimension and resolve score
        dimension = self._extract_dimension_from_prompt(prompt)
        score = self._dimension_scores.get(dimension, self._default_score)

        # Map score to 5-level: level = round(score * 4) + 1
        level = round(score * 4) + 1
        level = max(1, min(5, level))

        reasoning = f"Mock evaluation for dimension '{dimension}': level {level}/5"
        evidence = [
            EvidenceItem(
                modality="visual",
                timestamp=0.0,
                detail=f"Mock: scored {score:.2f}",
            )
        ]
        suggestion = f"Mock suggestion for {dimension}" if score < 0.75 else ""
        raw_output = json.dumps(
            {"level": level, "reasoning": reasoning, "evidence": [], "suggestion": suggestion}
        )

        return VLMResult(
            score=score,
            reasoning=reasoning,
            evidence=evidence,
            suggestion=suggestion,
            raw_output=raw_output,
        )

    def _extract_dimension_from_prompt(self, prompt: str) -> str:
        """Extract dimension name from rendered prompt text."""
        prompt_lower = prompt.lower()
        for dim in self._KNOWN_DIMENSIONS:
            if dim in prompt_lower or dim.replace("_", " ") in prompt_lower:
                return dim
        return "unknown"
