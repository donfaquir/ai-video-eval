# Phase 4 Spec 02：ProductFidelity 评估器

## 目标

实现商品还原度评估器，计算视频帧与商家主图的图文相似度，评估视频是否忠实展示了商品。

## 产出文件

- `video_eval/evaluators/product_fidelity.py`（新建）

## 类定义

```python
@register_evaluator("product_fidelity")
class ProductFidelityEvaluator(BaseEvaluator):
    name = "product_fidelity"
    version = "0.1.0"
    device_requirement = "gpu"
    requires = ["frames", "clip_features", "product_info"]
    default_weights = None
    config_schema = {
        "similarity_threshold": {"type": "float", "default": 0.3},
    }
```

## 实现要点

### __enter__

```python
def __enter__(self):
    import torch
    import open_clip
    
    # Reuse same SigLIP model as clip_features extractor for encoding product images
    # (clip_features already encoded video frames; we encode product images here)
    model_name = "ViT-SO400M-14-SigLIP-384"
    try:
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="webli"
        )
    except Exception:
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
    
    self._model = self._model.to(self.device_manager.device)
    self._model.eval()
    self._torch = torch
    return self
```

### __exit__

```python
def __exit__(self, *args):
    self._model = None
    self._preprocess = None
    if self._torch:
        if self.device_manager.device_type == "cuda":
            self._torch.cuda.empty_cache()
        elif self.device_manager.device_type == "mps" and hasattr(self._torch, "mps"):
            self._torch.mps.empty_cache()
```

### evaluate(context)

```python
def evaluate(self, context) -> EvalResult:
    clip_features = context.clip_features   # (num_frames, embed_dim) tensor from extractor
    product_info = context.product_info
    
    if clip_features is None or product_info is None:
        return EvalResult(
            dimension="product_fidelity", evaluator="product_fidelity",
            score=0.0, status="skipped", reason="missing_dependency",
        )
    
    # Encode product images
    product_features = self._encode_product_images(product_info.main_image_paths)
    if product_features is None:
        return EvalResult(
            dimension="product_fidelity", evaluator="product_fidelity",
            score=0.0, status="error", reason="evaluation_failed",
            evidence={"error": "Failed to encode product images"},
        )
    
    # Compute cosine similarity: each product image vs all video frames
    # similarity matrix: (num_products, num_frames)
    similarity = self._torch.nn.functional.cosine_similarity(
        product_features.unsqueeze(1),  # (P, 1, D)
        clip_features.unsqueeze(0),     # (1, F, D)
        dim=-1,
    )  # (P, F)
    
    # For each product image, find best matching frame
    best_per_product = []
    for i, img_path in enumerate(product_info.main_image_paths):
        best_frame_idx = similarity[i].argmax().item()
        best_sim = similarity[i].max().item()
        best_per_product.append({
            "product_image": img_path,
            "best_frame_idx": int(best_frame_idx),
            "best_frame_timestamp": context.frames[best_frame_idx].timestamp if context.frames else 0.0,
            "similarity": round(best_sim, 4),
        })
    
    # Overall score: average of best similarities across all product images
    avg_similarity = sum(m["similarity"] for m in best_per_product) / len(best_per_product) if best_per_product else 0.0
    # Normalize to [0, 1] (cosine similarity for CLIP is typically in [0.1, 0.5] range)
    # Map: 0.15 → 0.0, 0.45 → 1.0 (linear)
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
```

### _encode_product_images

```python
def _encode_product_images(self, image_paths: list[str]):
    """Load and encode product images. Returns (N, embed_dim) tensor or None."""
    from PIL import Image
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(self._preprocess(img))
        except Exception:
            continue
    
    if not images:
        return None
    
    batch = self._torch.stack(images).to(self.device_manager.device)
    with self._torch.no_grad():
        features = self._model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu()
```

## 关键行为

| 场景 | 行为 |
|------|------|
| product_info 缺失 | F3 自动跳过（requires 包含 product_info） |
| clip_features 缺失（CPU 机器） | F3 跳过 |
| 主图文件不存在 | 跳过该图，用其余图计算 |
| 全部主图加载失败 | status="error", reason="evaluation_failed" |
| 正常情况 | 计算相似度 → score ∈ [0,1] |

## 验收标准

- [ ] `video-eval plugins` 列出 product_fidelity（device=gpu, requires=[frames,clip_features,product_info]）
- [ ] GPU 机器 + 有商品信息：返回 score + evidence（product_matches 列表）
- [ ] CPU 机器：被 F2 跳过
- [ ] 无 product_info：被 F3 跳过（missing_product_info）
- [ ] evidence 包含每张主图的最佳匹配帧 + 相似度
