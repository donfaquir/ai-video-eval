"""ASR extractor using faster-whisper for speech-to-text transcription."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

from video_eval.core.base import BaseExtractor
from video_eval.core.registry import register_extractor
from video_eval.core.schemas import AsrResult

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext

logger = logging.getLogger(__name__)


@register_extractor("asr")
class ASRExtractor(BaseExtractor):
    """Extract speech transcription from video audio track via faster-whisper."""

    name = "asr"
    provides = ["asr"]
    requires: list[str] = []
    criticality = "optional"
    device_requirement = "any"
    config_schema = {
        "model_size": {"type": "str", "default": "large-v3"},
        "language": {"type": "str", "default": "auto"},
        "beam_size": {"type": "int", "default": 5},
    }

    def __enter__(self) -> ASRExtractor:
        """Load WhisperModel with lazy import (heavy dependency, section 2.1 convention)."""
        from faster_whisper import WhisperModel

        model_size = self.config.get("model_size", "large-v3")
        device = "cuda" if self.device_manager.device_type == "cuda" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self._model: Any = WhisperModel(model_size, device=device, compute_type=compute_type)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release model reference for GC."""
        self._model = None

    def extract(self, context: ReadonlyEvalContext) -> dict:
        """Transcribe audio from video.

        Returns:
            Dict with key "asr" containing an AsrResult instance.
            If no audio track is present, returns an empty AsrResult (not a failure).
        """
        video_path = context.video_path

        # Check has_audio from video metadata
        if context.video_meta and not context.video_meta.has_audio:
            return {"asr": AsrResult(full_text="", segments=[], language="none")}

        # Extract audio via ffmpeg to temp WAV
        audio_path = self._extract_audio(video_path)
        if audio_path is None:
            # ffmpeg failed to extract audio (no audio stream)
            return {"asr": AsrResult(full_text="", segments=[], language="none")}

        # Transcribe
        try:
            language_cfg = self.config.get("language", "auto")
            segments, info = self._model.transcribe(
                audio_path,
                language=None if language_cfg == "auto" else language_cfg,
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
            self._cleanup_audio(audio_path)

    def _extract_audio(self, video_path: str) -> str | None:
        """Extract audio to temp WAV file via ffmpeg.

        Returns:
            Path to the temporary WAV file, or None if extraction failed.
        """
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                audio_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            # Clean up the empty temp file on failure
            self._cleanup_audio(audio_path)
            return None
        return audio_path

    def _cleanup_audio(self, audio_path: str) -> None:
        """Remove temporary audio file if it exists."""
        try:
            os.unlink(audio_path)
        except OSError:
            pass
