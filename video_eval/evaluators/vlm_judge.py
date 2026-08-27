"""VLM-based multi-dimension evaluator with soft dependencies and failure budget."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from video_eval.core.base import BaseEvaluator
from video_eval.core.exceptions import VLMOutputParseError
from video_eval.core.registry import backend_registry, register_evaluator
from video_eval.core.schemas import EvalResult

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext, VLMResult

logger = logging.getLogger(__name__)

# Directory containing prompt template files
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@register_evaluator("vlm_judge")
class VLMJudge(BaseEvaluator):
    """Multi-dimension VLM evaluator with backend lifecycle management.

    Manages a VLM backend lifecycle, builds adaptive prompts based on
    available modalities, and enforces a failure budget across dimensions.
    """

    name = "vlm_judge"
    version = "0.1.0"
    device_requirement = "any"
    backend_config_key = "backend"
    requires = ["frames"]
    dimension_slots = {
        "main_image": ["sellpoint_coverage", "cross_modal"],
        "external": ["hook_strength", "marketing_logic", "audience_match", "cross_modal"],
        "general": ["cross_modal"],
    }
    default_weights = {
        "sellpoint_coverage": 0.20,
        "cross_modal": 0.15,
        "hook_strength": 0.20,
        "marketing_logic": 0.15,
        "audience_match": 0.10,
    }
    config_schema = {
        "backend": {"type": "str", "default": "mock", "required": True},
        "api_concurrency": {"type": "int", "default": 4},
        "api_max_failures": {"type": "int", "default": 5},
        "dimensions_main_image": {
            "type": "list",
            "default": ["sellpoint_coverage", "cross_modal"],
        },
        "dimensions_external": {
            "type": "list",
            "default": [
                "hook_strength",
                "marketing_logic",
                "audience_match",
                "cross_modal",
            ],
        },
        "dimensions_general": {"type": "list", "default": ["cross_modal"]},
    }

    def __enter__(self) -> VLMJudge:
        """Instantiate and enter the configured backend; load prompt templates."""
        backend_name = self.config.get("backend", "mock")
        backends_config = self.config.get("_backends", {})
        backend_cfg = backends_config.get(backend_name, {})

        BackendCls = backend_registry.get(backend_name)
        self._backend = BackendCls(self.device_manager, backend_cfg)
        self._backend.__enter__()

        # Load prompt templates from disk
        self._templates = self._load_templates()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release backend resources."""
        if hasattr(self, "_backend") and self._backend is not None:
            self._backend.__exit__(exc_type, exc_val, exc_tb)
            self._backend = None

    def evaluate(self, context: ReadonlyEvalContext) -> list[EvalResult]:
        """Evaluate all active dimensions with soft-dep checks and failure budget."""
        dimensions = self.slots_for(context.video_type)

        # Soft dependency checks
        asr_available = (
            context.asr is not None
            and "asr" not in context.extraction_failures
        )
        ocr_available = (
            bool(context.ocr)
            and "ocr" not in context.extraction_failures
        )

        results: list[EvalResult] = []
        failure_count = 0
        max_failures = self.config.get("api_max_failures", 5)

        for dim in dimensions:
            # Failure budget exceeded: placeholder for remaining dimensions
            if failure_count >= max_failures:
                results.append(self._placeholder(dim, reason="evaluation_failed"))
                continue

            # Sub-dimension dependency: sellpoint_coverage needs product_info
            if dim == "sellpoint_coverage" and context.product_info is None:
                results.append(
                    self._placeholder(dim, reason="missing_product_info")
                )
                continue

            # Sub-dimension dependency: cross_modal needs ASR OR OCR
            if dim == "cross_modal" and not asr_available and not ocr_available:
                results.append(
                    self._placeholder(dim, reason="missing_dependency")
                )
                continue

            # Build prompt with adaptive modality sections
            prompt = self._build_prompt(dim, context, asr_available, ocr_available)

            # Call backend
            try:
                vlm_result = self._backend.judge(context, prompt)
                result = self._convert(dim, vlm_result, asr_available, ocr_available)
                results.append(result)
            except VLMOutputParseError:
                failure_count += 1
                results.append(self._placeholder(dim, reason="parse_failed"))
            except Exception:
                failure_count += 1
                results.append(
                    self._placeholder(dim, reason="evaluation_failed")
                )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_templates(self) -> dict[str, str]:
        """Load all prompt templates from the prompts directory."""
        templates: dict[str, str] = {}
        # All known dimensions across all video types
        all_dims = set()
        for dims in self.dimension_slots.values():
            all_dims.update(dims)

        for dim in all_dims:
            template_path = _PROMPTS_DIR / f"{dim}.txt"
            if template_path.exists():
                templates[dim] = template_path.read_text(encoding="utf-8")
            else:
                logger.warning(
                    "Prompt template not found for dimension '%s': %s",
                    dim,
                    template_path,
                )
                templates[dim] = f"Evaluate {dim} for this {{video_type}} video."

        return templates

    def _build_prompt(
        self,
        dim: str,
        context: ReadonlyEvalContext,
        asr_available: bool,
        ocr_available: bool,
    ) -> str:
        """Render a prompt template with context-aware placeholder substitution."""
        template = self._templates.get(dim, f"Evaluate {dim}.")
        rendered = template

        # Replace basic placeholders
        rendered = rendered.replace("{video_type}", context.video_type)

        # Product info placeholders (sellpoint_coverage)
        if dim == "sellpoint_coverage" and context.product_info:
            rendered = rendered.replace(
                "{product_title}", context.product_info.title
            )
            rendered = rendered.replace(
                "{selling_points}",
                "\n".join(
                    f"- {sp}" for sp in context.product_info.selling_points
                ),
            )

        # Conditional ASR section
        if asr_available and context.asr:
            rendered = rendered.replace("{asr_text}", context.asr.full_text)
            # Keep ASR section content but remove markers
            rendered = re.sub(
                r"\{\{#if asr_available\}\}\n?", "", rendered
            )
            rendered = re.sub(
                r"\{\{/if\}\}",
                "",
                rendered,
                count=1,  # Remove first {{/if}} (belongs to ASR block)
            )
        else:
            # Remove entire ASR section between markers
            rendered = self._remove_section(rendered, "asr_available")

        # Conditional OCR section
        if ocr_available and context.ocr:
            ocr_text = "\n".join(item.text for item in context.ocr)
            rendered = rendered.replace("{ocr_text}", ocr_text)
            # Keep OCR section content but remove markers
            rendered = re.sub(
                r"\{\{#if ocr_available\}\}\n?", "", rendered
            )
            rendered = re.sub(r"\{\{/if\}\}", "", rendered, count=1)
        else:
            rendered = self._remove_section(rendered, "ocr_available")

        # Modality note
        modalities = ["visual"]
        if asr_available:
            modalities.append("audio/speech")
        if ocr_available:
            modalities.append("on-screen text")
        rendered = rendered.replace("{modality_note}", ", ".join(modalities))

        return rendered

    @staticmethod
    def _remove_section(text: str, section_name: str) -> str:
        """Remove a {{#if section_name}}...{{/if}} block from template text."""
        pattern = (
            r"\{\{#if " + re.escape(section_name) + r"\}\}.*?\{\{/if\}\}\n?"
        )
        return re.sub(pattern, "", text, count=1, flags=re.DOTALL)

    def _convert(
        self,
        dim: str,
        vlm_result: VLMResult,
        asr_available: bool,
        ocr_available: bool,
    ) -> EvalResult:
        """Convert a VLMResult into a standardized EvalResult."""
        input_modalities = ["visual"]
        if asr_available:
            input_modalities.append("asr")
        if ocr_available:
            input_modalities.append("ocr")

        return EvalResult(
            dimension=dim,
            evaluator="vlm_judge",
            score=vlm_result.score,
            status="scored",
            evidence={
                "vlm_evidence": [e.model_dump() for e in vlm_result.evidence],
                "input_modalities": input_modalities,
            },
            reasoning=vlm_result.reasoning,
            suggestion=vlm_result.suggestion,
        )

    @staticmethod
    def _placeholder(dim: str, reason: str) -> EvalResult:
        """Create an error/skipped placeholder result for a dimension."""
        status = (
            "error"
            if reason in ("evaluation_failed", "parse_failed")
            else "skipped"
        )
        return EvalResult(
            dimension=dim,
            evaluator="vlm_judge",
            score=0.0,
            status=status,
            reason=reason,
        )
