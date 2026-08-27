# Phase 2 Spec 05：抽取器编排增强 + 集成测试

## 目标

增强 Pipeline 中抽取器编排的边缘情况处理，补充 provides 冲突检测，实现端到端集成测试验证全部抽取器协同工作。

## 依赖

Spec 01-04 全部完成。

## 修改文件

- `video_eval/core/pipeline.py`（小幅增强）
- `video_eval/core/registry.py`（provides 冲突检测）
- `tests/test_extractors.py`（新建）
- `tests/test_integration_phase2.py`（新建）

## 1. provides 冲突检测

在 extractor_registry 的 register 时，检查新抽取器的 provides 是否与已注册抽取器冲突：

```python
# In register_extractor decorator or Registry.register for extractor_registry:
# After normal registration succeeds, check provides overlap
for existing_name in extractor_registry.list():
    if existing_name == name:
        continue
    existing_meta = extractor_registry.get_meta(existing_name)
    overlap = set(existing_meta.provides) & set(cls.provides)
    if overlap:
        raise DuplicateRegistrationError(
            f"Extractor '{name}' provides {overlap} which conflicts with '{existing_name}'",
            "extractor",
        )
```

注意：这个检查放在 `register_extractor` 装饰器内（registry.register 之后），而非 Registry 泛型类中（因为只有 extractor 有 provides 语义）。

## 2. Pipeline _run_extractors 增强

Phase 1 已实现核心逻辑。本 spec 确保：
- 抽取器 `__enter__` 失败时 extraction_failures 记录的 message 包含原始异常信息
- 对未识别的 criticality 值（非 "required"/"optional"）按 optional 处理并 warning

## 3. 集成测试

### tests/test_extractors.py

```python
"""Unit tests for extractors."""

def test_video_meta_basic(tmp_video_path):
    """video_meta extracts metadata and frames from a real video."""

def test_video_meta_no_audio(tmp_video_no_audio):
    """video_meta correctly reports has_audio=False."""

def test_video_meta_corrupt_file(tmp_path):
    """video_meta raises on corrupt video file."""

def test_asr_with_audio(tmp_video_with_speech):
    """ASR produces non-empty transcription."""

def test_asr_no_audio(tmp_video_no_audio):
    """ASR returns empty AsrResult when no audio track."""

def test_ocr_with_text(tmp_video_with_text):
    """OCR detects text in frames."""

def test_ocr_no_text(tmp_video_path):
    """OCR returns empty list for frames without text."""

def test_clip_features_shape(tmp_video_path):
    """CLIP features tensor has correct shape (num_frames, embed_dim)."""

def test_clip_features_normalized(tmp_video_path):
    """CLIP features are L2 normalized."""
```

### tests/test_integration_phase2.py

```python
"""End-to-end integration tests for Phase 2 extractors."""

def test_full_pipeline_general(tmp_video_path):
    """
    Run full pipeline with all extractors enabled.
    Verify: video_meta + asr + ocr all produce results.
    clip_features only if GPU available.
    """

def test_compliance_with_asr(tmp_video_with_keywords):
    """
    Video with speech containing limit words.
    ASR extracts text → compliance detects violations → score=0.0.
    """

def test_topo_sort_closure():
    """
    Only compliance enabled (requires asr, ocr).
    Verify: video_meta included via closure (ocr requires frames).
    """

def test_extraction_failure_degradation():
    """
    Mock ASR extractor failure.
    Verify: compliance gets skipped with reason=extraction_failed.
    Other evaluators unaffected.
    """
```

### conftest.py fixtures

```python
@pytest.fixture
def tmp_video_path(tmp_path):
    """Generate a 3-second 720p test video with audio."""
    # ffmpeg -f lavfi -i testsrc=duration=3:size=1280x720 -f lavfi -i sine=duration=3 ...

@pytest.fixture
def tmp_video_no_audio(tmp_path):
    """Generate a video without audio track."""

@pytest.fixture
def tmp_video_with_text(tmp_path):
    """Generate a video with visible text overlay."""
    # ffmpeg drawtext filter
```

## 验收标准

- [ ] provides 冲突：两个抽取器 provides 同一字段 → DuplicateRegistrationError
- [ ] `video-eval plugins --type extractor` 列出 4 个抽取器（video_meta/asr/ocr/clip_features）
- [ ] 端到端：`video-eval eval --video test.mp4 --video-type general --config phase2.yaml` 输出含 asr/ocr 数据
- [ ] 闭包扩展：只启用 compliance 时，video_meta 仍被自动纳入
- [ ] ASR 降级：asr 失败时 compliance `skipped: extraction_failed`（evidence 携带原因）
- [ ] 测试通过：`pytest tests/test_extractors.py tests/test_integration_phase2.py`
