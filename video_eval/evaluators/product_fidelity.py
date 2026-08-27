"""Product fidelity evaluator computing video-product visual similarity."""

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


@register_evaluator("product_fidelity")
class ProductFidelityEvaluator(BaseEvaluator):
    """Evaluate how faithfully a video represents the product.

    Computes cosine similarity between video frame embeddings (from clip_features
    extractor) and product main images encoded with the same SigLIP model.
    """

    name = "product_fidelity"
    version = "0.1.0"
    device_requirement = "gpu"
    requires = ["frames", "clip_features", "product_info"]
    default_weights = None
    config_schema = {
        "similarity_threshold": {"type": "float", "default": 0.3},
    }

    def __init__(self, device_manager: DeviceManager, config: dict) -> None:
        """Store references only. No heavy resource loading."""
        super().__init__(device_manager, config)
        self._torch: Any = None
        self._model: Any = None
        self._preprocess: Any = None

    def __enter__(self) -> ProductFidelityEvaluator:
        """Load SigLIP model (same as clip_features extractor) for encoding product images."""
        import torch
        import open_clip

        self._torch = torch

        model_name = "ViT-SO400M-14-SigLIP-384"

        # Try "webli" pretrained first (SigLIP), fall back to alternatives
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="webli"
            )
        except Exception as first_err:
            for alt_pretrained in ("openai", "laion2b_s34b_b79k"):
                try:
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        model_name, pretrained=alt_pretrained
                    )
                    logger.info(
                        "Loaded '%s' with pretrained='%s' (webli unavailable).",
                        model_name, alt_pretrained,
                    )
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to load model '{model_name}' with any pretrained weights. "
                    f"Original error: {first_err}"
                ) from first_err

        # On MPS, force float32 for safety
        if self.device_manager.device_type == "mps":
            model = model.float()

        model = model.to(self.device_manager.device)
        model.eval()

        self._model = model
        self._preprocess = preprocess

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release model resources and clear GPU cache."""
        self._model = None
        self._preprocess = None
        if self._torch:
            if self.device_manager.device_type == "cuda":
                self._torch.cuda.empty_cache()
            elif self.device_manager.device_type == "mps":
                if hasattr(self._torch, "mps") and hasattr(self._torch.mps, "empty_cache"):
                    self._torch.mps.empty_cache()

    def evaluate(self, context: ReadonlyEvalContext) -> EvalResult:
        """Compute video-product fidelity via cosine similarity.

        Returns:
            EvalResult with normalized score and per-product-image match evidence.
        """
        clip_features = context.clip_features
        product_info = context.product_info

        if clip_features is None or product_info is None:
            return EvalResult(
                dimension="product_fidelity",
                evaluator="product_fidelity",
                score=0.0,
                status="skipped",
                reason="missing_dependency",
            )

        if not product_info.main_image_paths:
            return EvalResult(
                dimension="product_fidelity",
                evaluator="product_fidelity",
                score=0.0,
                status="skipped",
                reason="no_product_images",
            )

        # Encode product images
        product_features = self._encode_product_images(product_info.main_image_paths)
        if product_features is None:
            return EvalResult(
                dimension="product_fidelity",
                evaluator="product_fidelity",
                score=0.0,
                status="error",
                reason="evaluation_failed",
                evidence={"error": "Failed to encode product images"},
            )

        # Ensure clip_features is on CPU float for computation
        frame_features = clip_features.float()
        product_features = product_features.float()

        # Compute cosine similarity: each product image vs all video frames
        # product_features: (P, D), frame_features: (F, D)
        # similarity matrix: (P, F)
        similarity = self._torch.nn.functional.cosine_similarity(
            product_features.unsqueeze(1),  # (P, 1, D)
            frame_features.unsqueeze(0),    # (1, F, D)
            dim=-1,
        )  # (P, F)

        # For each product image, find best matching frame
        best_per_product: list[dict] = []
        frames = context.frames
        for i, img_path in enumerate(product_info.main_image_paths):
            if i >= similarity.shape[0]:
                break
            best_frame_idx = similarity[i].argmax().item()
            best_sim = similarity[i].max().item()
            best_timestamp = (
                frames[best_frame_idx].timestamp
                if frames and best_frame_idx < len(frames)
                else 0.0
            )
            best_per_product.append({
                "product_image": img_path,
                "best_frame_idx": int(best_frame_idx),
                "best_frame_timestamp": best_timestamp,
                "similarity": round(best_sim, 4),
            })

        # Overall score: average of best similarities across all product images
        avg_similarity = (
            sum(m["similarity"] for m in best_per_product) / len(best_per_product)
            if best_per_product
            else 0.0
        )

        # Normalize to [0, 1]: CLIP cosine sim typically in [0.1, 0.5] range
        # Map: 0.15 -> 0.0, 0.45 -> 1.0 (linear)
        score = max(0.0, min(1.0, (avg_similarity - 0.15) / 0.30))

        return EvalResult(
            dimension="product_fidelity",
            evaluator="product_fidelity",
            score=score,
            status="scored",
            evidence={
                "product_matches": best_per_product,
                "avg_similarity": round(avg_similarity, 4),
                "num_product_images": len(product_info.main_image_paths),
            },
        )

    def _encode_product_images(self, image_paths: list[str]) -> Any:
        """Load and encode product images.

        Returns:
            Tensor of shape (N, embed_dim) L2-normalized, or None if all fail.
        """
        from PIL import Image

        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(self._preprocess(img))
            except Exception as exc:
                logger.warning("Failed to load product image '%s': %s", path, exc)
                continue

        if not images:
            return None

        device = self.device_manager.device
        batch = self._torch.stack(images).to(device)

        # On MPS, ensure float32
        if self.device_manager.device_type == "mps":
            batch = batch.float()

        with self._torch.no_grad():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu()
