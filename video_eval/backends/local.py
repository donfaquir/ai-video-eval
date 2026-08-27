"""Local backend for Qwen3-VL inference on GPU."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from video_eval.backends._parsing import parse_vlm_output
from video_eval.core.base import BaseBackend
from video_eval.core.registry import register_backend
from video_eval.core.schemas import VLMResult

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext


@register_backend("local")
class LocalBackend(BaseBackend):
    """Local VLM backend using Qwen3-VL for GPU inference."""

    name = "local"
    version = "0.1.0"
    device_requirement = "gpu"
    config_schema: dict = {
        "model": {"type": "str", "default": "Qwen/Qwen3-VL-8B-Instruct"},
        "max_new_tokens": {"type": "int", "default": 1024},
    }

    def __enter__(self) -> LocalBackend:
        """Load Qwen2.5-VL model and processor onto GPU device."""
        # Lazy import heavy dependencies
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "torch is not installed. "
                "Install it with: pip install torch"
            )

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError:
            raise RuntimeError(
                "transformers package is not installed or does not support Qwen2.5-VL. "
                "Install it with: pip install transformers>=4.37"
            )

        model_name = self.config.get("model", "Qwen/Qwen3-VL-8B-Instruct")
        self._max_new_tokens: int = self.config.get("max_new_tokens", 1024)

        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.device_manager.dtype,
            device_map="auto",
        )

        # D8: frame tensor cache (capacity=1)
        self._frame_cache: dict[str, Any] = {}
        self._torch = torch

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release model resources and clear GPU cache."""
        self._model = None
        self._processor = None
        self._frame_cache.clear()

        if self._torch and self.device_manager.device_type == "cuda":
            self._torch.cuda.empty_cache()
        elif self._torch and self.device_manager.device_type == "mps":
            if hasattr(self._torch, "mps") and hasattr(self._torch.mps, "empty_cache"):
                self._torch.mps.empty_cache()

    def judge(self, context: ReadonlyEvalContext, prompt: str) -> VLMResult:
        """Judge video quality using local Qwen3-VL model.

        Uses D8 frame cache to avoid re-preprocessing frames for the same video.
        """
        frames_input = self._get_or_preprocess(context)
        messages = self._build_messages(frames_input, prompt)

        # Tokenize and generate
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move inputs to model device
        device = self._model.device if hasattr(self._model, "device") else None
        if device is not None:
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
        )

        # Decode only the generated tokens (skip the input tokens)
        input_len = inputs.get("input_ids", self._torch.tensor([])).shape[-1]
        generated_ids = outputs[0][input_len:]
        raw_output = self._processor.decode(generated_ids, skip_special_tokens=True)

        return parse_vlm_output(raw_output)

    # ------------------------------------------------------------------
    # D8 frame cache (capacity=1)
    # ------------------------------------------------------------------

    def _get_or_preprocess(self, context: ReadonlyEvalContext) -> list[Any]:
        """Return cached preprocessed frames or compute and cache them."""
        video_path = context.video_path
        if video_path in self._frame_cache:
            return self._frame_cache[video_path]

        # Clear old entry (capacity=1)
        self._frame_cache.clear()

        # Preprocess frames from context
        processed = self._preprocess_frames(context.frames)
        self._frame_cache[video_path] = processed
        return processed

    def _preprocess_frames(self, frames: list | None) -> list[Any]:
        """Convert FrameItem list to PIL Images for the processor.

        Extracts the image field from each FrameItem. If frames are numpy
        arrays, converts them to PIL Images.
        """
        if not frames:
            return []

        from PIL import Image
        import numpy as np

        processed = []
        for frame_item in frames:
            img = frame_item.image
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            processed.append(img)
        return processed

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(self, frames: list[Any], prompt: str) -> list[dict]:
        """Build Qwen-VL chat messages with video frames and prompt.

        Constructs a multi-modal message in the format expected by
        Qwen2.5-VL's chat template.
        """
        # Build content list with interleaved images and text
        content: list[dict] = []

        # Add frames as images
        for frame in frames:
            content.append({"type": "image", "image": frame})

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "user", "content": content},
        ]
        return messages
