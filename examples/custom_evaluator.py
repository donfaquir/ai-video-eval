"""Example: Creating a custom evaluator plugin."""
from video_eval.core.base import BaseEvaluator
from video_eval.core.registry import register_evaluator
from video_eval.core.schemas import EvalResult


@register_evaluator("watermark_detect")
class WatermarkEvaluator(BaseEvaluator):
    """Detect watermarks in video frames."""

    name = "watermark_detect"
    version = "0.1.0"
    device_requirement = "any"
    requires = ["frames"]
    config_schema = {
        "sensitivity": {"type": "float", "default": 0.8},
    }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def evaluate(self, context):
        # Your detection logic here
        return EvalResult(
            dimension="watermark_detect",
            evaluator="watermark_detect",
            score=0.95,
            status="scored",
            evidence={"watermark_found": False},
        )
