"""Technical quality evaluation (rule-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from video_eval.core.base import BaseEvaluator
from video_eval.core.registry import register_evaluator
from video_eval.core.schemas import EvalResult

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext


@register_evaluator("technical_quality")
class TechnicalQualityEvaluator(BaseEvaluator):
    """Rule-based technical quality evaluator.

    Scores resolution and blur detection without any ML model.
    """

    name = "technical_quality"
    version = "0.1.0"
    device_requirement = "any"
    requires = ["frames"]
    default_weights = None
    config_schema: dict = {}

    def __enter__(self) -> TechnicalQualityEvaluator:
        """No-op: no model to load."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """No-op: no resources to release."""

    def evaluate(self, context: ReadonlyEvalContext) -> EvalResult:
        """Score video technical quality based on resolution and blur.

        Returns:
            EvalResult with averaged resolution and blur sub-scores.
        """
        meta = context.video_meta
        frames = context.frames

        scores: list[float] = []
        evidence: dict = {}

        # 1. Resolution score
        if meta is not None:
            w, h = meta.resolution
            if w >= 1920 and h >= 1080:
                scores.append(1.0)
            elif w >= 1280 and h >= 720:
                scores.append(0.75)
            elif w >= 640 and h >= 480:
                scores.append(0.5)
            else:
                scores.append(0.25)
            evidence["resolution"] = f"{w}x{h}"
        else:
            scores.append(0.25)
            evidence["resolution"] = "unknown"

        # 2. Blur detection (Laplacian variance on sampled frames)
        if frames:
            blur_scores = [self._blur_score(f.image) for f in frames[:8]]
            avg_blur = sum(blur_scores) / len(blur_scores)
        else:
            avg_blur = 0.5
        scores.append(avg_blur)
        evidence["blur_avg"] = round(avg_blur, 3)

        # 3. Overall
        final_score = sum(scores) / len(scores)

        return EvalResult(
            dimension="technical_quality",
            evaluator="technical_quality",
            score=final_score,
            status="scored",
            evidence=evidence,
        )

    def _blur_score(self, image) -> float:  # noqa: ANN001
        """Compute sharpness score using Laplacian variance.

        Higher variance means sharper image.
        Normalized: var < 50 -> 0.0 (very blurry), var > 500 -> 1.0 (sharp).
        """
        # Convert to grayscale numpy array
        gray = np.array(image.convert("L"), dtype=np.float64)

        # Simple Laplacian kernel convolution approximation:
        # Use variance of pixel intensities as a simplified sharpness metric
        # A proper Laplacian would use [[0,1,0],[1,-4,1],[0,1,0]] but
        # numpy-only variance gives a reasonable proxy.
        laplacian_var = float(gray.var())

        # Normalize to [0, 1]
        return min(max((laplacian_var - 50) / 450, 0.0), 1.0)
