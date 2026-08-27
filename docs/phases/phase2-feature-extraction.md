# Phase 2：特征抽取层

## 目标

实现全部 4 个内置抽取器（video_meta / asr / ocr / clip_features），Mac MPS 上可运行 GPU 抽取器。所有评估器的特征依赖在本阶段就绪。

## 前置依赖

Phase 1 完成（Registry + Pipeline + BaseExtractor + video_meta 基础实现）。

## 交付范围

### 1. ASR 抽取器（faster-whisper）

```python
@register_extractor("asr")
class ASRExtractor(BaseExtractor):
    name = "asr"
    provides = ["asr"]
    requires = []              # 自行从视频文件提取音频
    criticality = "optional"
    device_requirement = "any"
```

实现要点：
- `__enter__` 内 import faster_whisper（重依赖延迟 import）
- 从 video_path 提取音频流（ffmpeg subprocess）
- 转写为 AsrResult（full_text + segments + language）
- 视频无音轨时返回 `AsrResult(full_text="", segments=[], language="none")`（合法空值，不是失败）
- 配置项：model_size / language / beam_size（extractors.asr 段）

### 2. OCR 抽取器

```python
@register_extractor("ocr")
class OCRExtractor(BaseExtractor):
    name = "ocr"
    provides = ["ocr"]
    requires = ["frames"]      # 消费 video_meta 产出的 frames
    criticality = "optional"
    device_requirement = "any"
```

实现要点：
- `__enter__` 内 import easyocr 或 paddleocr
- 逐帧检测文字（按采样策略，不需要每帧都跑）
- 输出 `list[OcrItem]`（frame_idx + timestamp + text + bbox）
- 配置项：confidence 阈值（extractors.ocr 段）

### 3. CLIP 特征抽取器（SigLIP）

```python
@register_extractor("clip_features")
class CLIPFeaturesExtractor(BaseExtractor):
    name = "clip_features"
    provides = ["clip_features"]
    requires = ["frames"]      # 消费 frames
    criticality = "optional"
    device_requirement = "gpu"
```

实现要点：
- `__enter__` 内 import open_clip / torch
- 加载 ViT-SO400M-14-SigLIP-384 模型
- 对 frames 做 batch encode，输出特征 tensor
- MPS 兼容（dtype=float16）
- 配置项：model_name（extractors.clip_features 段）

### 4. video_meta 增强

Phase 1 已有基础实现，本阶段增强：
- 转场检测（帧间差异大于阈值）
- 智能抽帧策略（均匀 + 关键帧补充）
- has_audio 检测
- 配置项：fps / max_frames（extractors 顶层共享键）

### 5. 抽取器编排增强

- **sequential 模式下的 chunk 级抽取**：抽取器按 chunk 重新 enter/exit
- **批量模式下外层 enter 失败**：按 criticality 定性，整批共享（详设 §4.1 B7）
- **provides 冲突检测**：两个抽取器 provides 同一字段 → 注册拒绝

### 6. `extractors.<name>.enabled` 开关

- 支持 `enabled: false` 禁用抽取器
- 禁用后其 provides 字段不可用，依赖评估器被 F3 跳过
- strict_veto 依赖的抽取器被禁用 → 退出码 2

### 7. 测试

| 测试 | 覆盖点 |
|------|--------|
| `test_extractors.py` | 拓扑排序/依赖闭包扩展/required 失败中断/optional 降级/merge 校验/provides 冲突/enabled 开关 |
| `test_asr.py` | faster-whisper 集成（需 optional dep）/无音轨处理/segments 格式 |
| `test_ocr.py` | OCR 集成/空帧处理/confidence 过滤 |
| `test_clip.py` | SigLIP 加载/GPU↔CPU 回退/tensor 形状校验 |

## 验收标准

- [ ] 默认配置下 4 个抽取器全部注册（plugins 命令可见）
- [ ] GPU 机器：clip_features 正常产出 tensor
- [ ] CPU 机器：clip_features 被 F2 设备过滤跳过，不崩溃
- [ ] 未安装 faster-whisper：asr 字段级降级，compliance `skipped: extraction_failed`
- [ ] `video-eval eval --video test.mp4 --video-type general`：video_meta + ocr 正常产出
- [ ] 拓扑排序：只启用 compliance 时，video_meta 仍被闭包扩展纳入（为 ocr 提供 frames）
- [ ] sequential batch 模式：抽取器按 chunk 正确 enter/exit

## 参考详设章节

§2.3（BaseExtractor）、§4.5（抽取器编排全节）、§6.6（内置抽取器总表）、§7.3（extractors 配置段）
