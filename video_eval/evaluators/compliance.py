"""Compliance evaluation (rule-based keyword matching)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from video_eval.core.base import BaseEvaluator
from video_eval.core.registry import register_evaluator
from video_eval.core.schemas import EvalResult

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext


@register_evaluator("compliance")
class ComplianceEvaluator(BaseEvaluator):
    """Rule-based compliance evaluator.

    Checks ASR and OCR text against configurable word lists.
    Any violation results in score=0.0 (veto behavior).
    """

    name = "compliance"
    version = "0.1.0"
    device_requirement = "any"
    requires = ["asr", "ocr"]
    default_weights = None
    config_schema = {
        "limit_words": {"type": "list", "default": []},
        "medical_words": {"type": "list", "default": []},
        "banned_entities": {"type": "list", "default": []},
    }

    def __enter__(self) -> ComplianceEvaluator:
        """No-op: no model to load."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """No-op: no resources to release."""

    def evaluate(self, context: ReadonlyEvalContext) -> EvalResult:
        """Check text content against compliance word lists.

        Returns:
            EvalResult with score=1.0 if clean, score=0.0 if any violation found.
        """
        violations: list[dict] = []

        # Collect all text sources: (source_name, text)
        texts: list[tuple[str, str]] = []
        if context.asr and context.asr.full_text:
            texts.append(("asr", context.asr.full_text))
        if context.ocr:
            for item in context.ocr:
                texts.append(("ocr", item.text))

        # Check limit words
        limit_words = self.config.get("limit_words", [])
        for source, text in texts:
            for word in limit_words:
                if word in text:
                    violations.append(
                        {"type": "limit_word", "word": word, "source": source}
                    )

        # Check medical words
        medical_words = self.config.get("medical_words", [])
        for source, text in texts:
            for word in medical_words:
                if word in text:
                    violations.append(
                        {"type": "medical_word", "word": word, "source": source}
                    )

        # Check banned entities
        banned_entities = self.config.get("banned_entities", [])
        for source, text in texts:
            for entity in banned_entities:
                if entity in text:
                    violations.append(
                        {"type": "banned_entity", "word": entity, "source": source}
                    )

        score = 0.0 if violations else 1.0
        return EvalResult(
            dimension="compliance",
            evaluator="compliance",
            score=score,
            status="scored",
            evidence={"violations": violations, "total_violations": len(violations)},
        )
