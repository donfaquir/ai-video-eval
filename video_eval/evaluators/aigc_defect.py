"""AIGC defect detection evaluator using CLIP zero-shot classification."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from video_eval.core.base import BaseEvaluator
from video_eval.core.registry import register_evaluator
from video_eval.core.schemas import EvalResult

if TYPE_CHECKING:
    from video_eval.core.device import DeviceManager
    from video_eval.core.schemas import ReadonlyEvalContext

logger = logging.getLogger(__name__)


@register_evaluator("aigc_defect")
class AIGCDefectEvaluator(BaseEvaluator):
    """Detect AI generation artifacts in video frames via CLIP zero-shot.

    Uses CLIP ViT-L-14 to classify each frame as defective or normal based
    on text prompt similarity. Score = 1.0 means no defects detected.
    """

    name = "aigc_defect"
    version = "0.1.0"
    device_requirement = "gpu"
    requires = ["frames"]
    default_weights = None
    config_schema = {
        "model": {"type": "str", "default": "openai/clip-vit-large-patch14"},
        "defect_threshold": {"type": "float", "default": 0.6},
    }

    def __init__(self, device_manager: DeviceManager, config: dict) -> None:
        """Store references only. No heavy resource loading."""
        super().__init__(device_manager, config)
        self._torch: Any = None
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._text_features: Any = None

    def __enter__(self) -> AIGCDefectEvaluator:
        """Load CLIP ViT-L-14 model and pre-compute text features for zero-shot."""
        import torch
        import open_clip

        self._torch = torch

        # Load CLIP model (independent from clip_features extractor's SigLIP)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )

        # On MPS, force float32 for safety (MPS has incomplete float16 support)
        if self.device_manager.device_type == "mps":
            self._model = self._model.float()

        self._model = self._model.to(self.device_manager.device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer("ViT-L-14")

        # Pre-compute text features for zero-shot classification
        self._defect_prompts = [
            "a distorted image with visual artifacts",
            "an image with unnatural deformations",
            "a blurry corrupted AI-generated image",
            "an image with twisted warped objects",
            "unrealistic body proportions",
        ]
        self._normal_prompts = [
            "a normal natural photograph",
            "a clear high quality image",
            "a realistic scene without artifacts",
        ]
        self._text_features = self._encode_text_prompts()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release model resources and clear GPU cache."""
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None
        if self._torch:
            if self.device_manager.device_type == "cuda":
                self._torch.cuda.empty_cache()
            elif self.device_manager.device_type == "mps":
                if hasattr(self._torch, "mps") and hasattr(self._torch.mps, "empty_cache"):
                    self._torch.mps.empty_cache()

    def evaluate(self, context: ReadonlyEvalContext) -> EvalResult:
        """Score video frames for AIGC defects via zero-shot classification.

        Returns:
            EvalResult with score = 1.0 - avg_defect_probability.
        """
        frames = context.frames
        if not frames:
            return EvalResult(
                dimension="aigc_defect",
                evaluator="aigc_defect",
                score=0.0,
                status="skipped",
                reason="no_frames",
            )

        threshold = self.config.get("defect_threshold", 0.6)

        defect_frames: list[dict] = []
        total_defect_score = 0.0

        for frame_item in frames:
            # Encode frame
            image_features = self._encode_image(frame_item.image)
            # Compute similarity with defect vs normal prompts
            defect_prob = self._compute_defect_probability(image_features)

            if defect_prob >= threshold:
                defect_frames.append({
                    "frame_idx": frame_item.frame_idx,
                    "timestamp": frame_item.timestamp,
                    "defect_prob": round(defect_prob, 3),
                })
            total_defect_score += defect_prob

        # Score: 1.0 = no defects, 0.0 = all frames defective
        avg_defect = total_defect_score / len(frames)
        score = max(0.0, min(1.0, 1.0 - avg_defect))

        return EvalResult(
            dimension="aigc_defect",
            evaluator="aigc_defect",
            score=score,
            status="scored",
            evidence={
                "defect_frames": defect_frames,
                "total_frames_analyzed": len(frames),
                "frames_with_defects": len(defect_frames),
                "avg_defect_probability": round(avg_defect, 3),
            },
        )

    def _encode_text_prompts(self) -> Any:
        """Encode defect and normal prompts into a single text feature tensor.

        Returns:
            Tensor of shape (num_defect + num_normal, embed_dim), L2-normalized.
            First num_defect rows are defect prompts, rest are normal prompts.
        """
        all_prompts = self._defect_prompts + self._normal_prompts
        tokens = self._tokenizer(all_prompts).to(self.device_manager.device)

        with self._torch.no_grad():
            text_features = self._model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _encode_image(self, image: Any) -> Any:
        """Encode a single PIL image into CLIP feature space.

        Returns:
            L2-normalized image feature tensor of shape (1, embed_dim).
        """
        device = self.device_manager.device
        img_tensor = self._preprocess(image).unsqueeze(0).to(device)

        # On MPS, ensure float32
        if self.device_manager.device_type == "mps":
            img_tensor = img_tensor.float()

        with self._torch.no_grad():
            image_features = self._model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def _compute_defect_probability(self, image_features: Any) -> float:
        """Compute probability that image has AIGC defects via zero-shot.

        Uses softmax over average defect similarity vs average normal similarity.
        """
        # Cosine similarity with all text prompts: (1, num_prompts)
        similarities = (image_features @ self._text_features.T).squeeze(0)

        num_defect = len(self._defect_prompts)
        defect_sims = similarities[:num_defect]
        normal_sims = similarities[num_defect:]

        # Average similarity per category
        avg_defect_sim = defect_sims.mean()
        avg_normal_sim = normal_sims.mean()

        # Softmax over [defect, normal] to get probability
        logits = self._torch.stack([avg_defect_sim, avg_normal_sim])
        # Scale logits (CLIP similarities are typically small; scale up for sharper softmax)
        probs = self._torch.nn.functional.softmax(logits * 100.0, dim=0)

        # defect probability is the first element
        return probs[0].item()
