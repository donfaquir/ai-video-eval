"""CLIP visual feature extractor using open_clip (SigLIP)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from video_eval.core.base import BaseExtractor
from video_eval.core.registry import register_extractor

if TYPE_CHECKING:
    from video_eval.core.device import DeviceManager
    from video_eval.core.schemas import ReadonlyEvalContext

logger = logging.getLogger(__name__)


@register_extractor("clip_features")
class CLIPFeaturesExtractor(BaseExtractor):
    """Extract CLIP visual features from video frames.

    Uses open_clip with SigLIP model by default. Produces an L2-normalized
    feature tensor of shape (num_frames, embed_dim).
    """

    name = "clip_features"
    provides = ["clip_features"]
    requires = ["frames"]
    criticality = "optional"
    device_requirement = "gpu"
    config_schema = {
        "model_name": {"type": "str", "default": "ViT-SO400M-14-SigLIP-384"},
        "batch_size": {"type": "int", "default": 8},
    }

    def __init__(self, device_manager: DeviceManager, config: dict) -> None:
        """Store references only. No heavy resource loading."""
        super().__init__(device_manager, config)
        self._torch: Any = None
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def __enter__(self) -> CLIPFeaturesExtractor:
        """Load open_clip model and move to device."""
        import torch
        import open_clip

        self._torch = torch

        model_name = self.config.get("model_name", "ViT-SO400M-14-SigLIP-384")
        device = self.device_manager.device

        # Try "webli" pretrained first (SigLIP), fall back to auto-detect
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained="webli",
            )
        except Exception:
            logger.info(
                "Failed to load '%s' with pretrained='webli', "
                "falling back to default pretrained.",
                model_name,
            )
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained="",
            )

        # On MPS, force float32 for safety (MPS has incomplete float16 support)
        if self.device_manager.device_type == "mps":
            model = model.float()

        model = model.to(device)
        model.eval()

        self._model = model
        self._preprocess = preprocess

        try:
            self._tokenizer = open_clip.get_tokenizer(model_name)
        except Exception:
            # Tokenizer is not strictly needed for image encoding
            self._tokenizer = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release model resources and clear GPU cache."""
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        if self._torch and self.device_manager.device_type == "cuda":
            self._torch.cuda.empty_cache()

    def extract(self, context: ReadonlyEvalContext) -> dict:
        """Encode frames with CLIP model and return L2-normalized features.

        Returns:
            Dict with key "clip_features" containing a tensor of shape
            (num_frames, embed_dim), or None if no frames available.
        """
        frames = context.frames
        if not frames:
            return {"clip_features": None}

        batch_size = self.config.get("batch_size", 8)
        device = self.device_manager.device
        device_type = self.device_manager.device_type

        all_features = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]

            # Preprocess PIL Images -> stacked tensor
            images = self._torch.stack([
                self._preprocess(frame_item.image) for frame_item in batch_frames
            ])

            # On MPS, keep float32; on CUDA use autocast for efficiency
            if device_type == "mps":
                images = images.to(device, dtype=self._torch.float32)
            else:
                images = images.to(device)

            with self._torch.no_grad():
                if device_type == "cuda":
                    with self._torch.amp.autocast(device_type="cuda"):
                        features = self._model.encode_image(images)
                else:
                    features = self._model.encode_image(images)

                # L2 normalize
                features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu().float())

        # Concatenate all batches -> (num_frames, embed_dim)
        clip_features = self._torch.cat(all_features, dim=0)

        return {"clip_features": clip_features}
