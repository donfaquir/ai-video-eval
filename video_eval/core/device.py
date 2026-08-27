"""Device detection and management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class DeviceManager:
    """Device detection and adaptation.

    Provides device auto-detection, dtype selection, memory queries,
    and device-requirement matching for model plugins.

    torch is an optional dependency. When not installed, the manager
    falls back to cpu with device/dtype properties returning None.
    """

    def __init__(self, preferred: str | None = None) -> None:
        """Initialize DeviceManager with optional device preference.

        Args:
            preferred: Force a specific device type ("cuda", "mps", "cpu").
                       None means auto-detect with priority: cuda > mps > cpu.
        """
        self._torch: Any = None

        try:
            import torch as _torch

            self._torch = _torch
        except ImportError:
            pass

        # Determine device type
        if preferred:
            self._device_type = preferred
        elif self._torch and self._torch.cuda.is_available():
            self._device_type = "cuda"
        elif (
            self._torch
            and hasattr(self._torch.backends, "mps")
            and self._torch.backends.mps.is_available()
        ):
            self._device_type = "mps"
        else:
            self._device_type = "cpu"

        # Create torch objects if available
        if self._torch:
            self._device: torch.device | None = self._torch.device(self._device_type)
            self._dtype: torch.dtype | None = (
                self._torch.float16
                if self._device_type != "cpu"
                else self._torch.float32
            )
        else:
            self._device = None
            self._dtype = None

    @property
    def device(self) -> "torch.device | None":
        """Return the torch.device instance, or None if torch is not available."""
        return self._device

    @property
    def dtype(self) -> "torch.dtype | None":
        """Return the torch.dtype instance, or None if torch is not available."""
        return self._dtype

    @property
    def device_type(self) -> str:
        """Return the device type string: 'cuda', 'mps', or 'cpu'."""
        return self._device_type

    def is_gpu(self) -> bool:
        """Return True if the device is a GPU (cuda or mps)."""
        return self._device_type in ("cuda", "mps")

    def memory_info(self) -> dict[str, float]:
        """Query available and total memory for the current device.

        Returns:
            Dictionary with "free_gb" and "total_gb" as float values.
            For cuda: uses torch.cuda.mem_get_info.
            For mps/cpu: uses psutil system memory (approximate).
        """
        if self._device_type == "cuda" and self._torch:
            free, total = self._torch.cuda.mem_get_info()
            return {"free_gb": free / (1024**3), "total_gb": total / (1024**3)}
        else:
            import psutil

            mem = psutil.virtual_memory()
            return {
                "free_gb": mem.available / (1024**3),
                "total_gb": mem.total / (1024**3),
            }

    def can_load_model(self, param_count: int) -> bool:
        """Estimate whether a model can fit in available memory.

        Uses the formula: param_count * 2 bytes (float16) with 30% headroom.

        Args:
            param_count: Number of model parameters.

        Returns:
            True if estimated memory requirement fits in free memory.
        """
        required_gb = (param_count * 2) / (1024**3) * 1.3
        return self.memory_info()["free_gb"] >= required_gb

    def satisfies(self, requirement: str) -> bool:
        """Check if the current device satisfies a device requirement.

        Matching matrix:
            - "any"  -> always True
            - "gpu"  -> True if cuda OR mps
            - "cuda" -> True only if cuda
            - "mps"  -> True only if mps

        Args:
            requirement: One of "any", "gpu", "cuda", "mps".

        Returns:
            True if the current device meets the requirement.
        """
        if requirement == "any":
            return True
        if requirement == "gpu":
            return self._device_type in ("cuda", "mps")
        return self._device_type == requirement
