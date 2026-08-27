# Phase 2 Spec 04：CLIP 特征抽取器

## 目标

实现 CLIP 特征抽取器，使用 SigLIP 模型对帧做视觉编码，产出特征 tensor。供 ProductFidelity（Phase 4）消费。

## 依赖

Spec 01（video_meta 增强完成，frames 可用）。

## 产出文件

- `video_eval/extractors/clip_features.py`（新建）

## 类定义

```python
@register_extractor("clip_features")
class CLIPFeaturesExtractor(BaseExtractor):
    name = "clip_features"
    provides = ["clip_features"]
    requires = ["frames"]          # 消费 video_meta 产出的 frames
    criticality = "optional"
    device_requirement = "gpu"     # 需要 GPU（cuda 或 mps）
    config_schema = {
        "model_name": {"type": "str", "default": "ViT-SO400M-14-SigLIP-384"},
        "batch_size": {"type": "int", "default": 8},
    }
```

## 实现要点

### __enter__

```python
def __enter__(self):
    import torch
    import open_clip
    
    model_name = self.config.get("model_name", "ViT-SO400M-14-SigLIP-384")
    
    # open_clip model loading
    self._model, _, self._preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained="webli",  # SigLIP pretrained weights
    )
    self._model = self._model.to(self.device_manager.device)
    self._model.eval()
    
    self._tokenizer = open_clip.get_tokenizer(model_name)
    self._torch = torch
    
    return self
```

### __exit__

```python
def __exit__(self, *args):
    # Release model and clear GPU cache
    self._model = None
    self._preprocess = None
    self._tokenizer = None
    if self._torch and self.device_manager.device_type == "cuda":
        self._torch.cuda.empty_cache()
```

### extract(context)

```python
def extract(self, context) -> dict:
    frames = context.frames
    if not frames:
        return {"clip_features": None}
    
    batch_size = self.config.get("batch_size", 8)
    device = self.device_manager.device
    
    all_features = []
    
    # Process frames in batches
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        
        # Preprocess: PIL Image → tensor
        images = self._torch.stack([
            self._preprocess(frame_item.image) for frame_item in batch_frames
        ]).to(device)
        
        # Encode
        with self._torch.no_grad(), self._torch.amp.autocast(device_type=self.device_manager.device_type):
            features = self._model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
        
        all_features.append(features.cpu())
    
    # Concatenate all batches → (num_frames, embed_dim) tensor
    clip_features = self._torch.cat(all_features, dim=0)
    
    return {"clip_features": clip_features}
```

### 关键行为

| 场景 | 行为 |
|------|------|
| 正常 GPU 机器 | 加载模型，batch 编码，返回 tensor |
| CPU 机器 | F2 device 过滤直接跳过（device_requirement="gpu"），不执行 |
| MPS Mac | device="mps"，正常运行（satisfies("gpu")=True） |
| open_clip 未安装 | `__enter__` ImportError → optional 降级 |
| 空帧列表 | 返回 clip_features=None |
| 模型下载失败 | `__enter__` 异常 → optional 降级 |

### 显存管理

- batch_size 控制单次 GPU 推理的帧数（默认 8）
- 每个 batch 结束后 features 移到 CPU
- `__exit__` 时释放模型并清 CUDA cache
- MPS 上无 empty_cache（跳过）

## 验收标准

- [ ] `video-eval plugins --type extractor` 列出 clip_features（device=gpu）
- [ ] MPS Mac：加载 SigLIP 成功，产出 tensor shape=(N, embed_dim)
- [ ] CPU 机器：clip_features 被设备过滤跳过，不崩溃
- [ ] 特征已 L2 归一化（norm ≈ 1.0）
- [ ] `__exit__` 后模型释放（内存回收）
- [ ] open_clip 未安装时：字段级降级
