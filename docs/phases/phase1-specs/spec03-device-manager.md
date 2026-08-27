# Spec 03：DeviceManager

## 目标

实现 `DeviceManager` 类，提供设备检测、dtype 选择、显存查询与适配性判定。所有插件通过它获知运行设备。

## 依赖

Spec 01（项目结构）。无需 schemas/exceptions。

## 产出文件

- `video_eval/core/device.py`

## 完整实现

参考详设 §2.5。

```python
class DeviceManager:
    """Device detection and adaptation."""

    def __init__(self, preferred: str | None = None) -> None:
        """
        Args:
            preferred: Force device. None = auto-detect (cuda > mps > cpu).
        """
        self._device: torch.device
        self._dtype: torch.dtype
        self._device_type: str  # "cuda" / "mps" / "cpu"
        # Detection logic:
        # 1. If preferred is given and valid, use it
        # 2. Else: cuda if available > mps if available > cpu
        # 3. dtype: float16 for cuda/mps, float32 for cpu

    @property
    def device(self) -> "torch.device": ...

    @property
    def dtype(self) -> "torch.dtype": ...

    @property
    def device_type(self) -> str: ...

    def is_gpu(self) -> bool:
        """True if cuda or mps."""
        return self._device_type in ("cuda", "mps")

    def memory_info(self) -> dict:
        """
        Returns {"free_gb": float, "total_gb": float}.
        - cuda: torch.cuda.mem_get_info
        - mps: psutil system memory (approximate, marked as such)
        - cpu: psutil system memory
        """
        ...

    def can_load_model(self, param_count: int) -> bool:
        """
        Estimate if model fits in memory.
        Formula: param_count * 2 bytes (float16) with 30% headroom.
        """
        required_gb = (param_count * 2) / (1024**3) * 1.3
        return self.memory_info()["free_gb"] >= required_gb

    def satisfies(self, requirement: str) -> bool:
        """
        Check if current device satisfies a device_requirement.

        Matching matrix:
        - "any"  -> always True
        - "gpu"  -> True if cuda OR mps
        - "cuda" -> True only if cuda
        - "mps"  -> True only if mps
        """
        if requirement == "any":
            return True
        if requirement == "gpu":
            return self._device_type in ("cuda", "mps")
        return self._device_type == requirement
```

### torch 延迟导入

torch 是可选依赖（`[gpu]`）。DeviceManager 必须在 **torch 未安装时仍能构造**（降级为 cpu）：

```python
def __init__(self, preferred: str | None = None):
    try:
        import torch
        self._torch = torch
    except ImportError:
        self._torch = None

    if preferred:
        self._device_type = preferred
    elif self._torch and self._torch.cuda.is_available():
        self._device_type = "cuda"
    elif self._torch and hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
        self._device_type = "mps"
    else:
        self._device_type = "cpu"

    if self._torch:
        self._device = self._torch.device(self._device_type)
        self._dtype = self._torch.float16 if self._device_type != "cpu" else self._torch.float32
    else:
        self._device = None  # type: ignore
        self._dtype = None   # type: ignore
```

### memory_info 实现细节

```python
def memory_info(self) -> dict:
    if self._device_type == "cuda" and self._torch:
        free, total = self._torch.cuda.mem_get_info()
        return {"free_gb": free / (1024**3), "total_gb": total / (1024**3)}
    else:
        # mps / cpu: use psutil
        import psutil
        mem = psutil.virtual_memory()
        return {"free_gb": mem.available / (1024**3), "total_gb": mem.total / (1024**3)}
```

## 依赖包

- `torch`（可选，`[gpu]`）
- `psutil`（加入 core dependencies）

## 验收标准

- [ ] `DeviceManager()` 在无 torch 环境下构造成功，device_type="cpu"
- [ ] `DeviceManager("cpu").satisfies("any")` → True
- [ ] `DeviceManager("cpu").satisfies("gpu")` → False
- [ ] `DeviceManager("cuda").satisfies("gpu")` → True
- [ ] `DeviceManager("mps").satisfies("gpu")` → True
- [ ] `DeviceManager("cuda").satisfies("mps")` → False
- [ ] `memory_info()` 返回合法 dict（free_gb > 0, total_gb > 0）
- [ ] `can_load_model(7_000_000_000)` 在 Mac 上返回 bool（不崩溃）
- [ ] `is_gpu()` 在 cpu 上返回 False
