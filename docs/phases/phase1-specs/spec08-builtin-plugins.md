# Spec 08：内置插件（最小集）

## 目标

实现 Phase 1 所需的最小插件集：video_meta 抽取器、TechnicalQuality 评估器、Compliance 评估器、MockBackend。验证全链路可跑通。

## 依赖

Spec 04（Registry 装饰器）、Spec 05（基类）、Spec 02（schemas）。

## 产出文件

- `video_eval/extractors/video_meta.py`
- `video_eval/evaluators/technical_quality.py`
- `video_eval/evaluators/compliance.py`
- `video_eval/backends/mock.py`

---

## 1. video_meta 抽取器

```python
@register_extractor("video_meta")
class VideoMetaExtractor(BaseExtractor):
    name = "video_meta"
    provides = ["video_meta", "frames"]
    requires = []
    criticality = "required"
    device_requirement = "any"
    config_schema = {
        "fps": {"type": "int", "default": 1},
        "max_frames": {"type": "int", "default": 64},
    }
```

### extract() 实现

```python
def extract(self, context: ReadonlyEvalContext) -> dict:
    video_path = context.video_path
    # 1. ffprobe: get resolution, duration, fps, bitrate, has_audio
    meta = self._probe(video_path)
    # 2. ffmpeg/imageio: extract frames at configured fps, cap at max_frames
    frames = self._extract_frames(video_path, meta.fps, meta.duration)
    return {"video_meta": meta, "frames": frames}
```

### 实现细节

- **ffprobe**：`subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path])`
  - resolution: video stream width × height
  - duration: float seconds
  - fps: r_frame_rate (eval fraction)
  - bitrate: bit_rate int
  - has_audio: any stream with codec_type=="audio"
- **抽帧**：`imageio` 或 `ffmpeg` subprocess 按 `1/fps` 间隔采样
  - 帧数 = min(duration * config_fps, max_frames)
  - 输出 `list[FrameItem]`（image 为 PIL.Image）
- **ffprobe/ffmpeg 不存在**：`__enter__` 中检查 `shutil.which("ffprobe")`，不存在抛异常（criticality=required → 中断）

---

## 2. TechnicalQuality 评估器

```python
@register_evaluator("technical_quality")
class TechnicalQualityEvaluator(BaseEvaluator):
    name = "technical_quality"
    version = "0.1.0"
    device_requirement = "any"
    requires = ["frames"]
    default_weights = None     # weights from config
    config_schema = {}
```

### evaluate() 实现

纯规则，不需要模型：

```python
def evaluate(self, context: ReadonlyEvalContext) -> EvalResult:
    meta = context.video_meta
    frames = context.frames

    scores = []
    evidence = {}

    # 1. Resolution score
    w, h = meta.resolution
    if w >= 1920 and h >= 1080:
        scores.append(1.0)
    elif w >= 1280 and h >= 720:
        scores.append(0.75)
    elif w >= 640 and h >= 480:
        scores.append(0.5)
    else:
        scores.append(0.25)
    evidence["resolution"] = f"{w}x{h}"

    # 2. Blur detection (Laplacian variance on sampled frames)
    blur_scores = [self._blur_score(f.image) for f in frames[:8]]
    avg_blur = sum(blur_scores) / len(blur_scores) if blur_scores else 0.5
    scores.append(avg_blur)
    evidence["blur_avg"] = round(avg_blur, 3)

    # 3. Overall
    final_score = sum(scores) / len(scores)
    return EvalResult(
        dimension="technical_quality",
        evaluator="technical_quality",
        score=final_score,
        status="scored",
        evidence=evidence,
    )
```

### _blur_score 辅助

```python
def _blur_score(self, image) -> float:
    """Laplacian variance normalized to [0, 1]. Higher = sharper."""
    import numpy as np
    arr = np.array(image.convert("L"))
    laplacian_var = arr.var()  # simplified; real impl uses cv2.Laplacian
    # Normalize: var < 50 = very blurry (0.0), var > 500 = sharp (1.0)
    return min(max((laplacian_var - 50) / 450, 0.0), 1.0)
```

---

## 3. Compliance 评估器

```python
@register_evaluator("compliance")
class ComplianceEvaluator(BaseEvaluator):
    name = "compliance"
    version = "0.1.0"
    device_requirement = "any"
    requires = ["asr", "ocr"]
    default_weights = None     # veto-only, no weight
    config_schema = {
        "limit_words": {"type": "list", "default": []},
        "medical_words": {"type": "list", "default": []},
        "banned_entities": {"type": "list", "default": []},
    }
```

### evaluate() 实现

```python
def evaluate(self, context: ReadonlyEvalContext) -> EvalResult:
    violations = []

    # Collect all text
    texts = []
    if context.asr and context.asr.full_text:
        texts.append(("asr", context.asr.full_text, context.asr.segments))
    if context.ocr:
        for item in context.ocr:
            texts.append(("ocr", item.text, [{"start": item.timestamp, "end": item.timestamp}]))

    # Check limit words
    for source, text, segments in texts:
        for word in self.config.get("limit_words", []):
            if word in text:
                violations.append({"type": "limit_word", "word": word, "source": source})

    # Check medical words
    for source, text, segments in texts:
        for word in self.config.get("medical_words", []):
            if word in text:
                violations.append({"type": "medical_word", "word": word, "source": source})

    score = 0.0 if violations else 1.0
    return EvalResult(
        dimension="compliance",
        evaluator="compliance",
        score=score,
        status="scored",
        evidence={"violations": violations, "total_violations": len(violations)},
    )
```

---

## 4. MockBackend

```python
@register_backend("mock")
class MockBackend(BaseBackend):
    name = "mock"
    version = "0.1.0"
    device_requirement = "any"
    config_schema = {}

    def __init__(self, device_manager, config):
        super().__init__(device_manager, config)
        self._default_score = config.get("default_score", 0.75)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def judge(self, context: ReadonlyEvalContext, prompt: str) -> VLMResult:
        return VLMResult(
            score=self._default_score,
            reasoning="Mock evaluation - fixed score",
            evidence=[],
            suggestion="",
            raw_output=f"mock: score={self._default_score}",
        )
```

---

## 验收标准

- [ ] `video-eval plugins` 列出 video_meta / technical_quality / compliance / mock（注册成功）
- [ ] video_meta 对有效视频文件返回 `{"video_meta": VideoMeta(...), "frames": [...]}`
- [ ] video_meta 对不存在文件抛异常（criticality=required → ExtractionError）
- [ ] TechnicalQuality 对帧列表返回合法 EvalResult（score ∈ [0,1]）
- [ ] Compliance 对含极限词的文本返回 score=0.0 + violations evidence
- [ ] Compliance 对空文本返回 score=1.0
- [ ] MockBackend.judge() 返回合法 VLMResult（固定分数）
- [ ] ffprobe 不存在时 video_meta.__enter__ 失败 → ExtractionError 上抛
