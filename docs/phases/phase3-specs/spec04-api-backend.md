# Phase 3 Spec 04：APIBackend

## 目标

实现 API 后端，支持 Gemini/OpenAI 等云端 VLM API 调用，含重试、超时、D8 上传缓存。

## 产出文件

- `video_eval/backends/api.py`（新建）

## 类定义

```python
@register_backend("api")
class APIBackend(BaseBackend):
    name = "api"
    version = "0.1.0"
    device_requirement = "any"
    config_schema = {
        "provider": {"type": "str", "default": "gemini"},
        "model": {"type": "str", "default": "gemini-2.0-flash"},
        "timeout": {"type": "int", "default": 30},
        "max_retries": {"type": "int", "default": 3},
        "retry_base": {"type": "float", "default": 1.0},
    }
```

## 实现要点

### __enter__

```python
def __enter__(self):
    self._provider = self.config.get("provider", "gemini")
    self._model = self.config.get("model", "gemini-2.0-flash")
    self._timeout = self.config.get("timeout", 30)
    self._max_retries = self.config.get("max_retries", 3)
    self._retry_base = self.config.get("retry_base", 1.0)
    
    # Validate API key from environment
    if self._provider == "gemini":
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
    elif self._provider == "openai":
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
    else:
        self._api_key = os.environ.get(f"{self._provider.upper()}_API_KEY", "")
    
    if not self._api_key:
        raise RuntimeError(f"API key not found for provider '{self._provider}'. "
                          f"Set {self._provider.upper()}_API_KEY environment variable.")
    
    # D8: upload cache (capacity=1, thread-safe)
    self._upload_cache: dict[str, str] = {}  # video_path → file_ref
    self._cache_lock = threading.Lock()
    
    return self
```

### __exit__

```python
def __exit__(self, *args):
    # Clear upload cache (API-side file deletion if needed)
    self._upload_cache.clear()
    self._api_key = None
```

### judge()

```python
def judge(self, context, prompt: str) -> VLMResult:
    # D8: reuse upload for same video_path
    video_ref = self._get_or_upload(context.video_path)
    
    # Call API with retry
    raw_output = self._call_with_retry(video_ref, prompt)
    
    # Parse output
    return self._parse_output(raw_output)
```

### D8 上传缓存（容量=1）

```python
def _get_or_upload(self, video_path: str) -> str:
    with self._cache_lock:
        if video_path in self._upload_cache:
            return self._upload_cache[video_path]
        # Capacity = 1: clear old entry if video_path changed
        if self._upload_cache:
            self._upload_cache.clear()
        ref = self._upload_video(video_path)
        self._upload_cache[video_path] = ref
        return ref
```

### 重试逻辑（指数退避）

```python
def _call_with_retry(self, video_ref: str, prompt: str) -> str:
    last_error = None
    for attempt in range(self._max_retries + 1):
        try:
            return self._call_api(video_ref, prompt)
        except Exception as exc:
            last_error = exc
            if attempt < self._max_retries:
                delay = self._retry_base * (2 ** attempt)
                time.sleep(delay)
    raise RuntimeError(f"API call failed after {self._max_retries} retries: {last_error}")
```

### 输出解析（五级映射）

```python
def _parse_output(self, raw_output: str) -> VLMResult:
    # Try to extract JSON from output
    # Pattern: find {...} block
    # Parse level → score = (level - 1) / 4
    # Raise VLMOutputParseError on failure
    ...
```

### Provider 抽象

```python
def _call_api(self, video_ref: str, prompt: str) -> str:
    if self._provider == "gemini":
        return self._call_gemini(video_ref, prompt)
    elif self._provider == "openai":
        return self._call_openai(video_ref, prompt)
    else:
        raise RuntimeError(f"Unknown provider: {self._provider}")
```

Gemini 和 OpenAI 各自用其 SDK 或 HTTP API 调用。Phase 3 至少实现 Gemini；OpenAI 可留 stub。

## 验收标准

- [ ] `video-eval plugins --type backend` 列出 api（device=any）
- [ ] 无 API key 时 __enter__ 抛 RuntimeError（明确提示设置环境变量）
- [ ] 有 GEMINI_API_KEY 时：端到端调用返回 VLMResult
- [ ] 重试逻辑：模拟 API 失败后指数退避重试
- [ ] D8 缓存：同一 video_path 连续调用只上传 1 次
- [ ] D8 缓存容量=1：video_path 变化时清空旧条目
- [ ] 输出解析：提取 JSON 中的 level 并映射为 [0,1] 分数
- [ ] 解析失败：抛 VLMOutputParseError（携带 raw_output 片段）
