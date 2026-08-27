"""Custom exception hierarchy for the video-eval framework."""

from __future__ import annotations


class VideoEvalError(Exception):
    """Base for all video-eval exceptions."""

    pass


# --- Registry exceptions ---


class RegistryError(VideoEvalError):
    """Base for Registry errors."""

    def __init__(self, message: str, registry_name: str = ""):
        self.registry_name = registry_name
        super().__init__(message)


class NameNotFoundError(RegistryError):
    """get() miss. Message contains suggestions."""

    pass


class RegistryFrozenError(RegistryError):
    """Attempted registration after freeze."""

    pass


class DuplicateRegistrationError(RegistryError):
    """Concrete->Concrete conflict without allow_override."""

    pass


class MaterializationError(RegistryError):
    """Lazy load (import or type check) failed."""

    pass


# --- Pipeline exceptions ---


class ExtractionError(VideoEvalError):
    """Required extractor failed. Carries extractor name."""

    def __init__(self, extractor_name: str):
        self.extractor_name = extractor_name
        super().__init__(f"Required extractor '{extractor_name}' failed")


class ConfigError(VideoEvalError):
    """Configuration validation failure (exit code 2)."""

    pass


# --- Backend exceptions ---


class VLMOutputParseError(VideoEvalError):
    """Backend failed to parse VLM output."""

    def __init__(self, message: str, raw_output: str = ""):
        self.raw_output = raw_output[:500]  # truncate for safety
        super().__init__(message)
