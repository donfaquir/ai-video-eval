"""API backend for cloud VLM services (Gemini, OpenAI)."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any

from video_eval.backends._parsing import parse_vlm_output
from video_eval.core.base import BaseBackend
from video_eval.core.registry import register_backend
from video_eval.core.schemas import VLMResult

if TYPE_CHECKING:
    from video_eval.core.schemas import ReadonlyEvalContext


@register_backend("api")
class APIBackend(BaseBackend):
    """Cloud VLM API backend supporting Gemini and OpenAI providers."""

    name = "api"
    version = "0.1.0"
    device_requirement = "any"
    config_schema: dict = {
        "provider": {"type": "str", "default": "gemini"},
        "model": {"type": "str", "default": "gemini-2.0-flash"},
        "timeout": {"type": "int", "default": 30},
        "max_retries": {"type": "int", "default": 3},
        "retry_base": {"type": "float", "default": 1.0},
    }

    def __enter__(self) -> APIBackend:
        """Validate API key, initialize provider client, and set up caches."""
        self._provider: str = self.config.get("provider", "gemini")
        self._model: str = self.config.get("model", "gemini-2.0-flash")
        self._timeout: int = self.config.get("timeout", 30)
        self._max_retries: int = self.config.get("max_retries", 3)
        self._retry_base: float = self.config.get("retry_base", 1.0)

        # Validate API key from environment
        env_var = self._get_api_key_env_var()
        self._api_key: str = os.environ.get(env_var, "")
        if not self._api_key:
            raise RuntimeError(
                f"API key not found for provider '{self._provider}'. "
                f"Set {env_var} environment variable."
            )

        # Initialize provider SDK client
        self._client: Any = self._init_client()

        # D8: upload cache (capacity=1, thread-safe)
        self._upload_cache: dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release resources and clear upload cache."""
        self._upload_cache.clear()
        self._api_key = ""
        self._client = None

    def judge(self, context: ReadonlyEvalContext, prompt: str) -> VLMResult:
        """Judge video quality via cloud VLM API.

        Uses D8 upload cache to avoid re-uploading the same video,
        and retries with exponential backoff on transient failures.
        """
        video_ref = self._get_or_upload(context.video_path)
        raw_output = self._call_with_retry(video_ref, prompt)
        return parse_vlm_output(raw_output)

    # ------------------------------------------------------------------
    # D8 upload cache (capacity=1)
    # ------------------------------------------------------------------

    def _get_or_upload(self, video_path: str) -> Any:
        """Return cached upload reference or upload video (capacity=1 eviction)."""
        with self._cache_lock:
            if video_path in self._upload_cache:
                return self._upload_cache[video_path]
            # Capacity = 1: clear old entry if video_path changed
            if self._upload_cache:
                self._upload_cache.clear()
            ref = self._upload_video(video_path)
            self._upload_cache[video_path] = ref
            return ref

    # ------------------------------------------------------------------
    # Retry logic (exponential backoff)
    # ------------------------------------------------------------------

    def _call_with_retry(self, video_ref: Any, prompt: str) -> str:
        """Call API with exponential backoff retry on failure."""
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._call_api(video_ref, prompt)
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = self._retry_base * (2 ** attempt)
                    time.sleep(delay)
        raise RuntimeError(
            f"API call failed after {self._max_retries} retries: {last_error}"
        )

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _get_api_key_env_var(self) -> str:
        """Return the environment variable name for the API key."""
        if self._provider == "gemini":
            return "GEMINI_API_KEY"
        elif self._provider == "openai":
            return "OPENAI_API_KEY"
        else:
            return f"{self._provider.upper()}_API_KEY"

    def _init_client(self) -> Any:
        """Initialize the provider SDK client."""
        if self._provider == "gemini":
            return self._init_gemini()
        elif self._provider == "openai":
            return self._init_openai()
        else:
            raise RuntimeError(f"Unknown provider: {self._provider}")

    def _call_api(self, video_ref: Any, prompt: str) -> str:
        """Dispatch API call to the appropriate provider."""
        if self._provider == "gemini":
            return self._call_gemini(video_ref, prompt)
        elif self._provider == "openai":
            return self._call_openai(video_ref, prompt)
        else:
            raise RuntimeError(f"Unknown provider: {self._provider}")

    def _upload_video(self, video_path: str) -> Any:
        """Upload video to provider storage."""
        if self._provider == "gemini":
            return self._upload_gemini(video_path)
        elif self._provider == "openai":
            # OpenAI does not have a separate upload step; return path directly
            return video_path
        else:
            raise RuntimeError(f"Unknown provider: {self._provider}")

    # ------------------------------------------------------------------
    # Gemini implementation
    # ------------------------------------------------------------------

    def _init_gemini(self) -> Any:
        """Initialize the Gemini SDK client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError(
                "google-generativeai package is not installed. "
                "Install it with: pip install google-generativeai"
            )

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model)
        return model

    def _upload_gemini(self, video_path: str) -> Any:
        """Upload a video file to Gemini File API."""
        import google.generativeai as genai

        # Upload file to Gemini for processing
        video_file = genai.upload_file(
            path=video_path,
            mime_type=self._guess_video_mime(video_path),
        )

        # Wait for file to become active
        # NOTE: For large files, upload_file may need polling via get_file()
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(
                f"Gemini file upload failed for: {video_path}"
            )

        return video_file

    def _call_gemini(self, video_ref: Any, prompt: str) -> str:
        """Call Gemini API with video and prompt."""
        response = self._client.generate_content(
            [video_ref, prompt],
            request_options={"timeout": self._timeout},
        )
        return response.text

    # ------------------------------------------------------------------
    # OpenAI implementation
    # ------------------------------------------------------------------

    def _init_openai(self) -> Any:
        """Initialize the OpenAI SDK client."""
        try:
            import openai
        except ImportError:
            raise RuntimeError(
                "openai package is not installed. "
                "Install it with: pip install openai"
            )

        client = openai.OpenAI(api_key=self._api_key)
        return client

    def _call_openai(self, video_ref: Any, prompt: str) -> str:
        """Call OpenAI API with video (base64) and prompt.

        NOTE: OpenAI's vision API primarily supports images.
        For video, we read the file and encode as base64.
        This may need adjustment based on OpenAI's video support evolution.
        """
        import base64

        # Read video file and encode as base64
        with open(video_ref, "rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")

        mime_type = self._guess_video_mime(video_ref)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:{mime_type};base64,{video_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            timeout=self._timeout,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_video_mime(video_path: str) -> str:
        """Guess MIME type from file extension."""
        ext = video_path.rsplit(".", 1)[-1].lower() if "." in video_path else ""
        mime_map = {
            "mp4": "video/mp4",
            "avi": "video/x-msvideo",
            "mov": "video/quicktime",
            "mkv": "video/x-matroska",
            "webm": "video/webm",
        }
        return mime_map.get(ext, "video/mp4")
