"""OCR extractor using easyocr for text detection in video frames."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from video_eval.core.base import BaseExtractor
from video_eval.core.registry import register_extractor
from video_eval.core.schemas import OcrItem

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext

logger = logging.getLogger(__name__)


@register_extractor("ocr")
class OCRExtractor(BaseExtractor):
    """Extract text from sampled video frames via easyocr."""

    name = "ocr"
    provides = ["ocr"]
    requires = ["frames"]
    criticality = "optional"
    device_requirement = "any"
    config_schema = {
        "confidence": {"type": "float", "default": 0.5},
        "languages": {"type": "list", "default": ["ch_sim", "en"]},
        "sample_interval": {"type": "int", "default": 3},
    }

    def __enter__(self) -> OCRExtractor:
        """Load easyocr Reader with lazy import (heavy dependency, section 2.1 convention)."""
        import easyocr

        languages = self.config.get("languages", ["ch_sim", "en"])
        use_gpu = self.device_manager.is_gpu()

        self._reader: Any = easyocr.Reader(languages, gpu=use_gpu)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release reader reference for GC."""
        self._reader = None

    def extract(self, context: ReadonlyEvalContext) -> dict:
        """Run OCR on sampled frames.

        Returns:
            Dict with key "ocr" containing a list of OcrItem instances.
            Empty frames or no text detected returns an empty list (not a failure).
        """
        import numpy as np

        frames = context.frames
        if not frames:
            return {"ocr": []}

        confidence_threshold = self.config.get("confidence", 0.5)
        sample_interval = self.config.get("sample_interval", 3)

        ocr_results: list[OcrItem] = []

        # Sample frames at interval (not every frame)
        sampled = frames[::sample_interval]

        for frame_item in sampled:
            try:
                # Convert PIL Image to numpy array for easyocr
                image_array = np.array(frame_item.image)
                detections = self._reader.readtext(image_array)

                for bbox, text, conf in detections:
                    if conf >= confidence_threshold:
                        # Flatten bbox to [x1, y1, x2, y2, ...]
                        flat_bbox = [float(coord) for point in bbox for coord in point]
                        ocr_results.append(OcrItem(
                            frame_idx=frame_item.frame_idx,
                            timestamp=frame_item.timestamp,
                            text=text,
                            bbox=flat_bbox,
                        ))
            except Exception as exc:
                # Skip frame on detection failure, continue with next
                logger.warning(
                    "OCR detection failed for frame %d: %s",
                    frame_item.frame_idx,
                    exc,
                )
                continue

        return {"ocr": ocr_results}
