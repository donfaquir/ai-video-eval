# Phase 3 Spec 05：LocalBackend

## 目标

实现本地 VLM 后端，使用 Qwen3-VL 模型做推理。需要 GPU 设备。

## 产出文件

- `video_eval/backends/local.py`（新建）

## 类定义

```python
@register_backend("local")
class LocalBackend(BaseBackend):
    name = "local"
    version = "0.1.0"
    device_requirement = "gpu"
    config_schema = {
        "model": {"type": "str", "default": "Qwen/Qwen3-VL-8B-Instruct"},
        "max_new_tokens": {"type": "int", "default": 1024},
    }
```

## 实现要点

### __enter__

```python
def __enter__(self):
    # Lazy import heavy dependencies
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    model_name = self.config.get("model", "Qwen/Qwen3-VL-8B-Instruct")
    device = self.device_manager.device
    
    self._processor = AutoProcessor.from_pretrained(model_name)
    self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=self.device_manager.dtype,
        device_map="auto",
    )
    
    # D8: frame tensor cache (capacity=1)
    self._frame_cache: dict[str, Any] = {}
    self._torch = torch
    
    return self
```

### __exit__

```python
def __exit__(self, *args):
    self._model = None
    self._processor = None
    self._frame_cache.clear()
    if self._torch and self.device_manager.device_type == "cuda":
        self._torch.cuda.empty_cache()
    elif self._torch and self.device_manager.device_type == "mps":
        if hasattr(self._torch, "mps") and hasattr(self._torch.mps, "empty_cache"):
            self._torch.mps.empty_cache()
```

### judge()

```python
def judge(self, context, prompt: str) -> VLMResult:
    # D8: cache preprocessed frames per video_path
    frames_input = self._get_or_preprocess(context)
    
    # Build messages for Qwen-VL format
    messages = self._build_messages(frames_input, prompt)
    
    # Generate
    inputs = self._processor.apply_chat_template(messages, ...)
    outputs = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
    raw_output = self._processor.decode(outputs[0], skip_special_tokens=True)
    
    # Parse
    return self._parse_output(raw_output)
```

### D8 帧缓存（容量=1）

```python
def _get_or_preprocess(self, context) -> list:
    video_path = context.video_path
    if video_path in self._frame_cache:
        return self._frame_cache[video_path]
    # Clear old (capacity=1)
    self._frame_cache.clear()
    # Preprocess frames for model input
    processed = self._preprocess_frames(context.frames)
    self._frame_cache[video_path] = processed
    return processed
```

### 输出解析

同 APIBackend：提取 JSON → level → `score = (level - 1) / 4`。共用解析逻辑可抽到公共模块。

## 注意事项

- 本 spec 依赖 GPU 设备 + 大模型下载（约 16GB）
- 在无 GPU 或无法下载模型的环境下，LocalBackend 会被 F2 设备过滤跳过
- 开发阶段可先用 MockBackend 测试 VLMJudge 全流程，LocalBackend 作为最后验证
- transformers 版本需支持 Qwen2.5-VL（>=4.37）

## 验收标准

- [ ] `video-eval plugins --type backend` 列出 local（device=gpu）
- [ ] CPU 机器：vlm_judge backend=local 被 F2 跳过
- [ ] GPU 机器：端到端加载 Qwen3-VL 并返回 VLMResult
- [ ] D8 缓存：同一视频多次 judge() 只预处理帧 1 次
- [ ] __exit__ 后模型释放、显存回收
- [ ] 输出解析：正确提取 5 级评分并映射到 [0,1]
