"""Video metadata and frame extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING

from video_eval.core.base import BaseExtractor
from video_eval.core.registry import register_extractor
from video_eval.core.schemas import FrameItem, VideoMeta

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext


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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed for '{video_path}': {result.stderr.strip()}"
            )

        data = json.loads(result.stdout)
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
        """Extract frames at configured fps, capped at max_frames."""
        from PIL import Image
        import io

        config_fps = self.config.get("fps", 1)
        max_frames = self.config.get("max_frames", 64)

        # Calculate number of frames to extract
        if meta.duration <= 0:
            num_frames = 1
        else:
            num_frames = min(int(meta.duration * config_fps), max_frames)
        num_frames = max(num_frames, 1)

        # Calculate interval between frames
        if meta.duration <= 0:
            interval = 0.0
        else:
            interval = meta.duration / num_frames

        frames: list[FrameItem] = []
        for i in range(num_frames):
            timestamp = i * interval
            # Use ffmpeg to extract a single frame at the given timestamp
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
            result = subprocess.run(
                cmd, capture_output=True, timeout=10
            )
            if result.returncode != 0 or not result.stdout:
                continue

            image = Image.open(io.BytesIO(result.stdout))
            frames.append(
                FrameItem(frame_idx=i, timestamp=timestamp, image=image)
            )

        return frames
