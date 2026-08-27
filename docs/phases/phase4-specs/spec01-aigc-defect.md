# Phase 4 Spec 01：AIGCDefect 评估器

## 目标

实现 AIGC 瑕疵检测评估器，使用 CLIP zero-shot 分类检测视频帧中的 AI 生成瑕疵（扭曲、变形、不自然）。

## 产出文件

- `video_eval/evaluators/aigc_defect.py`（新建）

## 类定义

```python
@register_evaluator("aigc_defect")
class AIGCDefectEvaluator(BaseEvaluator):
    name = "aigc_defect"
    version = "0.1.0"
    device_requirement = "gpu"
    requires = ["frames"]
    default_weights = None
    config_schema = {
        "model": {"type": "str", "default": "openai/clip-vit-large-patch14"},
        "defect_threshold": {"type": "float", "default": 0.6},
    }
```

## 实现要点

### __enter__

```python
def __enter__(self):
    import torch
    import open_clip
    
    model_name = self.config.get("model", "openai/clip-vit-large-patch14")
    # Load CLIP model (independent from clip_features extractor's SigLIP)
    # For open_clip: model_name="ViT-L-14", pretrained="openai"
    self._model, _, self._preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    self._model = self._model.to(self.device_manager.device)
    self._model.eval()
    self._tokenizer = open_clip.get_tokenizer("ViT-L-14")
    self._torch = torch
    
    # Pre-compute text features for zero-shot classification
    self._defect_prompts = [
        "a distorted image with visual artifacts",
        "an image with unnatural deformations",
        "a blurry or corrupted AI-generated image",
        "an image with twisted or warped objects",
        "an image with unrealistic body proportions",
    ]
    self._normal_prompts = [
        "a normal natural photograph",
        "a clear high quality image",
        "a realistic scene without artifacts",
    ]
    self._text_features = self._encode_text_prompts()
    return self
```

### __exit__

```python
def __exit__(self, *args):
    self._model = None
    self._preprocess = None
    self._tokenizer = None
    self._text_features = None
    if self._torch:
        if self.device_manager.device_type == "cuda":
            self._torch.cuda.empty_cache()
        elif self.device_manager.device_type == "mps" and hasattr(self._torch, "mps"):
            self._torch.mps.empty_cache()
```

### evaluate(context)

```python
def evaluate(self, context) -> EvalResult:
    frames = context.frames
    threshold = self.config.get("defect_threshold", 0.6)
    
    defect_frames = []
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
    avg_defect = total_defect_score / len(frames) if frames else 0.0
    score = max(0.0, 1.0 - avg_defect)
    
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
```

### _compute_defect_probability

```python
def _compute_defect_probability(self, image_features) -> float:
    """Compute probability that image has AIGC defects via zero-shot."""
    # Cosine similarity with defect and normal text embeddings
    # softmax over [defect_sim, normal_sim] → defect probability
    ...
```

## 验收标准

- [ ] `video-eval plugins` 列出 aigc_defect（device=gpu）
- [ ] GPU/MPS 机器：对视频帧返回 score ∈ [0,1]
- [ ] CPU 机器：被 F2 过滤跳过（device_unavailable）
- [ ] evidence 包含 defect_frames 列表（frame_idx + timestamp + defect_prob）
- [ ] defect_threshold 配置生效
- [ ] __exit__ 后模型释放
