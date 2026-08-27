"""Weighted-veto fusion strategy implementation."""

from __future__ import annotations

import logging

from video_eval.core.base import BaseFusion
from video_eval.core.registry import register_fusion
from video_eval.core.schemas import EvalResult, FusionOutcome

logger = logging.getLogger(__name__)


@register_fusion("weighted_veto")
class WeightedVetoFusion(BaseFusion):
    """Fusion strategy: veto scan + weighted score + grade determination."""

    name = "weighted_veto"
    version = "0.1.0"

    # Reasons that trigger strict_veto rejection (D5)
    STRICT_VETO_TRIGGERS = {
        "init_failed",
        "evaluation_failed",
        "parse_failed",
        "runtime_unavailable",
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.thresholds = config.get("thresholds", {"A": 0.75, "B": 0.60, "C": 0.40})
        self.veto_thresholds: dict[str, float] = config.get("veto_thresholds", {})
        self.strict_veto_dims: set[str] = set(config.get("strict_veto_dims", []))

    def fuse(
        self,
        results: dict[str, EvalResult],
        video_type: str,
        weights: dict,
        default_weights: dict[str, float | None],
    ) -> FusionOutcome:
        """Aggregate dimension scores into final outcome.

        5-step logic: veto scan -> weight calc -> grade -> suggestions -> return.
        """
        # ------------------------------------------------------------------
        # Step 1: Veto scan
        # ------------------------------------------------------------------
        veto_dims = set(self.veto_thresholds.keys()) | self.strict_veto_dims
        rejected = False
        veto_reasons: list[str] = []

        for name in veto_dims:
            result = results.get(name)
            if result is None:
                continue
            if result.status != "scored":
                if name in self.strict_veto_dims and result.reason in self.STRICT_VETO_TRIGGERS:
                    rejected = True
                    veto_reasons.append(
                        f"{name} execution failed ({result.reason}), "
                        f"strict veto dimension has no result"
                    )
                elif name in self.strict_veto_dims:
                    # Environment issue not caught by pre-check (should not be reachable)
                    logger.warning(
                        "Strict veto dimension '%s' has non-scored status '%s' "
                        "with reason '%s' which is not in STRICT_VETO_TRIGGERS; "
                        "not triggering veto.",
                        name,
                        result.status,
                        result.reason,
                    )
                continue
            threshold = self.veto_thresholds.get(name, 0.0)
            if result.score <= threshold:
                rejected = True
                veto_reasons.append(
                    f"{name} score {result.score:.2f} did not exceed "
                    f"veto threshold {threshold}"
                )

        # ------------------------------------------------------------------
        # Step 2: Weight calculation
        # ------------------------------------------------------------------
        scored_dims = [d for d, r in results.items() if r.status == "scored"]
        veto_only_dims = {
            d
            for d in veto_dims
            if weights.get(d) is None and default_weights.get(d) is None
        }
        weighted_dims = [d for d in scored_dims if d not in veto_only_dims]

        if rejected:
            overall_score = 0.0
        elif not weighted_dims:
            overall_score = 0.0
        else:
            raw_weights: dict[str, float] = {}
            for d in weighted_dims:
                w = weights.get(d)
                if w is None:
                    w = default_weights.get(d)
                if w is None:
                    logger.warning(
                        "Dimension %s has no weight (neither config nor "
                        "default_weights), treating as 0",
                        d,
                    )
                    w = 0.0
                raw_weights[d] = w
            total = sum(raw_weights.values())
            if total == 0:
                overall_score = 0.0
            else:
                normalized = {d: w / total for d, w in raw_weights.items()}
                overall_score = sum(
                    normalized[d] * results[d].score for d in weighted_dims
                )

        # ------------------------------------------------------------------
        # Step 3: Grade determination
        # ------------------------------------------------------------------
        if rejected:
            grade = "REJECT"
        elif overall_score >= self.thresholds["A"]:
            grade = "A"
        elif overall_score >= self.thresholds["B"]:
            grade = "B"
        elif overall_score >= self.thresholds["C"]:
            grade = "C"
        else:
            grade = "REJECT"

        # ------------------------------------------------------------------
        # Step 4: Suggestions
        # ------------------------------------------------------------------
        suggestion_threshold = self.thresholds["B"]
        suggestions = [
            f"[{d}] {results[d].suggestion}"
            for d in scored_dims
            if results[d].score < suggestion_threshold and results[d].suggestion
        ]

        # ------------------------------------------------------------------
        # Step 5: Return
        # ------------------------------------------------------------------
        return FusionOutcome(
            overall_score=overall_score,
            grade=grade,
            passed=(grade in ("A", "B")),
            veto_reasons=veto_reasons,
            suggestions=suggestions,
        )
