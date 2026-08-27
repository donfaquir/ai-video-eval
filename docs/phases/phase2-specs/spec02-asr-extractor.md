# Phase 2 Spec 02：ASR 抽取器

## 目标

实现 ASR 抽取器，使用 faster-whisper 对视频音轨做语音转写，产出 AsrResult。

## 依赖

Spec 01（video_meta 增强完成，has_audio 可用）。

## 产出文件

- `video_eval/extractors/asr.py`（新建）

## 类定义

```python
@register_extractor("asr")
class ASRExtractor(BaseExtractor):
    name = "asr"
    provides = ["asr"]
    requires = []                  # 自行从 video_path 提取音频，不依赖 frames
    criticality = "optional"
    device_requirement = "any"     # faster-whisper 可 CPU/GPU 运行
    config_schema = {
        "model_size": {"type": "str", "default": "large-v3"},
        "language": {"type": "str", "default": "auto"},
        "beam_size": {"type": "int", "default": 5},
    }
```

## 实现要点

### __enter__

```python
def __enter__(self):
    # Lazy import (heavy dependency, §2.1 convention)
    from faster_whisper import WhisperModel
    
    model_size = self.config.get("model_size", "large-v3")
    device = "cuda" if self.device_manager.device_type == "cuda" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return self
```

### __exit__

```python
def __exit__(self, *args):
    self._model = None  # Release model reference for GC
```

### extract(context)

```python
def extract(self, context) -> dict:
    video_path = context.video_path
    
    # 1. Check has_audio (if video_meta available)
    if context.video_meta and not context.video_meta.has_audio:
        # No audio track: return empty result (NOT a failure)
        return {"asr": AsrResult(full_text="", segments=[], language="none")}
    
    # 2. Extract audio via ffmpeg to temp WAV
    audio_path = self._extract_audio(video_path)
    if audio_path is None:
        # ffmpeg failed to extract audio (no audio stream)
        return {"asr": AsrResult(full_text="", segments=[], language="none")}
    
    # 3. Transcribe
    try:
        segments, info = self._model.transcribe(
            audio_path,
            language=None if self.config.get("language", "auto") == "auto" else self.config["language"],
            beam_size=self.config.get("beam_size", 5),
        )
        
        segment_list = []
        full_text_parts = []
        for seg in segments:
            segment_list.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())
        
        return {"asr": AsrResult(
            full_text=" ".join(full_text_parts),
            segments=segment_list,
            language=info.language or "unknown",
        )}
    finally:
        # Clean up temp audio file
        self._cleanup_audio(audio_path)
```

### _extract_audio 辅助

```python
def _extract_audio(self, video_path: str) -> str | None:
    """Extract audio to temp WAV file via ffmpeg. Returns path or None."""
    import tempfile, subprocess
    
    audio_path = tempfile.mktemp(suffix=".wav")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return audio_path
```

### 关键行为

| 场景 | 行为 |
|------|------|
| 视频有音轨 | 正常转写，返回 AsrResult |
| 视频无音轨（has_audio=False） | 返回空 AsrResult（full_text=""），**不是**失败 |
| faster-whisper 未安装 | `__enter__` ImportError → optional 降级 → extraction_failures["asr"] |
| ffmpeg 提取音频失败 | 返回空 AsrResult |
| 转写过程异常 | 抛出异常 → optional 降级 |

## 验收标准

- [ ] `video-eval plugins --type extractor` 列出 asr 抽取器
- [ ] 有音轨视频：产出非空 AsrResult（full_text 有内容）
- [ ] 无音轨视频：产出 AsrResult(full_text="", language="none")，不报错
- [ ] faster-whisper 未安装时：asr 字段级降级，compliance `skipped: extraction_failed`
- [ ] 转写结果 segments 格式正确（含 start/end/text）
