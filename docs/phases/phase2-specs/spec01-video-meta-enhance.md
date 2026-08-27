# Phase 2 Spec 01：video_meta 抽取器增强

## 目标

增强 Phase 1 的 video_meta 基础实现，补充转场检测、智能抽帧策略、has_audio 检测。

## 依赖

Phase 1 完成（video_eval/extractors/video_meta.py 已存在）。

## 修改文件

- `video_eval/extractors/video_meta.py`（增强已有实现）

## 增强内容

### 1. has_audio 检测

ffprobe 已返回 streams 信息，检查是否存在 `codec_type == "audio"` 的流。写入 `VideoMeta.has_audio`。

### 2. 智能抽帧策略

当前实现：均匀间隔采样。增强为：

```python
def _extract_frames(self, video_path, meta, config_fps, max_frames):
    """
    Strategy:
    1. Uniform sampling at config_fps (e.g., 1 fps)
    2. Total frames capped at max_frames
    3. Always include first and last frame
    """
```

确保：
- 首帧（timestamp=0）和尾帧（timestamp≈duration）必须包含
- 帧数 = min(ceil(duration * config_fps), max_frames)
- 返回按 timestamp 排序的 FrameItem 列表

### 3. 转场检测（帧间差异）

```python
def _detect_scene_changes(self, frames: list[FrameItem], threshold: float = 30.0) -> list[int]:
    """
    Compare adjacent frames by mean absolute pixel difference.
    Returns list of frame indices where scene change detected.
    Store in VideoMeta or as separate evidence (non-blocking, informational).
    """
```

将检测结果附加到 VideoMeta 或存为 evidence 字段（供后续评估器参考）。Phase 2 暂不强制要求下游消费。

### 4. 配置消费

读取 extractors 顶层共享键：
```python
config_fps = self.config.get("fps", 1)
max_frames = self.config.get("max_frames", 64)
```

### 5. 错误处理增强

- 视频文件损坏（ffprobe 返回非 0）→ 抛异常（criticality=required → 中断）
- 视频无视频流 → 抛异常
- 抽帧部分失败（某帧解码错误）→ 跳过该帧，记 warning

## 验收标准

- [ ] has_audio 对有音轨视频返回 True，对纯视频流返回 False
- [ ] 抽帧结果包含首帧和尾帧
- [ ] 帧数不超过 max_frames
- [ ] 损坏视频文件 → 明确异常（非静默空结果）
- [ ] `video-eval eval --video test.mp4 --video-type general` 输出含 video_meta.has_audio 字段
