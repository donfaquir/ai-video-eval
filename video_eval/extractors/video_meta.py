"""Video metadata and frame extraction."""

from __future__ import annotations

import io
import json
import logging
import math
import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from video_eval.core.base import BaseExtractor
from video_eval.core.registry import register_extractor
from video_eval.core.schemas import FrameItem, VideoMeta

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext

logger = logging.getLogger(__name__)


@register_extractor("video_meta")
class VideoMetaExtractor(BaseExtractor):
    """Extract video metadata via ffprobe and frames via ffmpeg."""

    name = "video_meta"
    provides = ["video_meta", "frames"]
    requires: list[str] = []
    criticality = "required"
    device_requirement = "any"
    config_schema = {
        "fps": {"type": "int", "default": 1},
        "max_frames": {"type": "int", "default": 64},
    }

    def __enter__(self) -> VideoMetaExtractor:
        """Verify ffprobe is available on the system."""
        if shutil.which("ffprobe") is None:
            raise RuntimeError(
                "ffprobe not found on PATH. Install ffmpeg to use VideoMetaExtractor."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """No-op: no resources to release."""

    def extract(self, context: ReadonlyEvalContext) -> dict:
        """Extract video metadata and frames.

        Returns:
            Dict with keys "video_meta" (VideoMeta) and "frames" (list[FrameItem]).
        """
        video_path = context.video_path
        meta = self._probe(video_path)
        frames = self._extract_frames(video_path, meta)
        scene_changes = self._detect_scene_changes(frames)
        meta = meta.model_copy(update={"scene_changes": scene_changes})
        return {"video_meta": meta, "frames": frames}

    def _probe(self, video_path: str) -> VideoMeta:
        """Run ffprobe and parse video metadata."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"ffprobe execution failed for '{video_path}': {exc}"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed for '{video_path}': {result.stderr.strip()}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"ffprobe returned invalid JSON for '{video_path}': {exc}"
            ) from exc

        streams = data.get("streams", [])
        fmt = data.get("format", {})

        # Find video stream
        video_stream = None
        has_audio = False
        for stream in streams:
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            if stream.get("codec_type") == "audio":
                has_audio = True

        if video_stream is None:
            raise RuntimeError(f"No video stream found in '{video_path}'")

        # Resolution
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Duration
        duration = float(
            video_stream.get("duration")
            or fmt.get("duration")
            or 0.0
        )

        # FPS from r_frame_rate (fraction like "30/1")
        r_frame_rate = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0

        # Bitrate
        bitrate = int(
            video_stream.get("bit_rate")
            or fmt.get("bit_rate")
            or 0
        )

        return VideoMeta(
            resolution=(width, height),
            duration=duration,
            fps=fps,
            bitrate=bitrate,
            has_audio=has_audio,
        )

    def _extract_frames(self, video_path: str, meta: VideoMeta) -> list[FrameItem]:
        """Extract frames with smart sampling strategy.

        Strategy:
        1. Uniform sampling at config_fps (e.g., 1 fps)
        2. Total frames capped at max_frames
        3. Always include first and last frame
        """
        config_fps = self.config.get("fps", 1)
        max_frames = self.config.get("max_frames", 64)

        # Calculate number of frames to extract
        if meta.duration <= 0:
            timestamps = [0.0]
        else:
            num_frames = min(math.ceil(meta.duration * config_fps), max_frames)
            num_frames = max(num_frames, 2)  # at least first + last

            if num_frames == 2:
                timestamps = [0.0, meta.duration]
            else:
                # Generate uniform timestamps, always including first and last
                # Interior frames are evenly spaced between 0 and duration
                interior_count = num_frames - 2
                interval = meta.duration / (interior_count + 1)
                timestamps = [0.0]
                for i in range(1, interior_count + 1):
                    timestamps.append(i * interval)
                timestamps.append(meta.duration)

        # Clamp last timestamp before end to avoid seek-past-end issues.
        # Use 1/fps margin (at least 0.1s) to ensure the last seek lands on a frame.
        if len(timestamps) > 1 and meta.duration > 0:
            margin = max(1.0 / meta.fps, 0.1) if meta.fps > 0 else 0.1
            timestamps[-1] = min(timestamps[-1], max(meta.duration - margin, 0.0))

        frames: list[FrameItem] = []
        for idx, timestamp in enumerate(timestamps):
            cmd = [
                "ffmpeg",
                "-ss", f"{timestamp:.3f}",
                "-i", video_path,
                "-vframes", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "-loglevel", "quiet",
                "pipe:1",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning(
                    "Frame extraction failed at timestamp %.3f for '%s': %s",
                    timestamp, video_path, exc,
                )
                continue

            if result.returncode != 0 or not result.stdout:
                logger.warning(
                    "Frame decode failed at timestamp %.3f for '%s' "
                    "(returncode=%d, output_size=%d)",
                    timestamp, video_path, result.returncode, len(result.stdout),
                )
                continue

            try:
                image = Image.open(io.BytesIO(result.stdout))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PIL failed to open frame at timestamp %.3f for '%s': %s",
                    timestamp, video_path, exc,
                )
                continue

            frames.append(
                FrameItem(frame_idx=idx, timestamp=timestamp, image=image)
            )

        return frames

    def _detect_scene_changes(
        self, frames: list[FrameItem], threshold: float = 30.0
    ) -> list[int]:
        """Compare adjacent frames by mean absolute pixel difference.

        Returns list of frame indices where scene change detected.
        Non-blocking: errors are logged and an empty list is returned.
        """
        if len(frames) < 2:
            return []

        scene_change_indices: list[int] = []
        try:
            prev_array = np.array(
                frames[0].image.convert("RGB"), dtype=np.float32
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scene change detection skipped: %s", exc)
            return []

        for i in range(1, len(frames)):
            try:
                curr_array = np.array(
                    frames[i].image.convert("RGB"), dtype=np.float32
                )
                # Resize if dimensions differ (shouldn't normally happen)
                if prev_array.shape != curr_array.shape:
                    curr_image = frames[i].image.convert("RGB").resize(
                        (prev_array.shape[1], prev_array.shape[0])
                    )
                    curr_array = np.array(curr_image, dtype=np.float32)

                diff = np.mean(np.abs(curr_array - prev_array))
                if diff > threshold:
                    scene_change_indices.append(frames[i].frame_idx)
                prev_array = curr_array
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Scene change detection error at frame %d: %s",
                    frames[i].frame_idx, exc,
                )
                continue

        return scene_change_indices
