# Phase 2 Spec 03：OCR 抽取器

## 目标

实现 OCR 抽取器，使用 easyocr 对采样帧做文字检测，产出 list[OcrItem]。

## 依赖

Spec 01（video_meta 增强完成，frames 可用）。

## 产出文件

- `video_eval/extractors/ocr.py`（新建）

## 类定义

```python
@register_extractor("ocr")
class OCRExtractor(BaseExtractor):
    name = "ocr"
    provides = ["ocr"]
    requires = ["frames"]          # 消费 video_meta 产出的 frames
    criticality = "optional"
    device_requirement = "any"     # easyocr 可 CPU 运行（GPU 加速可选）
    config_schema = {
        "confidence": {"type": "float", "default": 0.5},
        "languages": {"type": "list", "default": ["ch_sim", "en"]},
        "sample_interval": {"type": "int", "default": 3},
    }
```

## 实现要点

### __enter__

```python
def __enter__(self):
    import easyocr
    
    languages = self.config.get("languages", ["ch_sim", "en"])
    use_gpu = self.device_manager.is_gpu()
    
    self._reader = easyocr.Reader(languages, gpu=use_gpu)
    return self
```

### __exit__

```python
def __exit__(self, *args):
    self._reader = None
```

### extract(context)

```python
def extract(self, context) -> dict:
    frames = context.frames
    if not frames:
        return {"ocr": []}
    
    confidence_threshold = self.config.get("confidence", 0.5)
    sample_interval = self.config.get("sample_interval", 3)
    
    ocr_results = []
    
    # Sample frames at interval (not every frame)
    sampled = frames[::sample_interval]
    
    for frame_item in sampled:
        # Convert PIL Image to numpy array for easyocr
        import numpy as np
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
    
    return {"ocr": ocr_results}
```

### 关键行为

| 场景 | 行为 |
|------|------|
| 帧中无文字 | 返回空 list（合法值） |
| 帧列表为空 | 返回空 list |
| easyocr 未安装 | `__enter__` ImportError → optional 降级 |
| 单帧检测失败 | 跳过该帧，继续下一帧（不中断） |
| confidence 低于阈值 | 过滤掉该检测结果 |

### 采样策略

不是每帧都跑 OCR（太慢）：
- `sample_interval=3`：每 3 帧采样 1 帧（默认 1fps × 3 = 每 3 秒一帧做 OCR）
- 对于 60s 视频（64 帧上限），约检测 21 帧

## 验收标准

- [ ] `video-eval plugins --type extractor` 列出 ocr 抽取器
- [ ] 含文字帧的视频：产出 OcrItem 列表（text 非空）
- [ ] 纯画面无文字：产出空 list（不是 None，不报错）
- [ ] easyocr 未安装时：ocr 字段级降级
- [ ] OcrItem 包含 frame_idx / timestamp / text / bbox 四个字段
- [ ] confidence 过滤生效：低置信度检测被排除
