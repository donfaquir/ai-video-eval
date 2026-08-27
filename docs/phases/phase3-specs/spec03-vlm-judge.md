# Phase 3 Spec 03：VLMJudge 评估器

## 目标

实现 VLMJudge 多维度评估器，管理 backend 生命周期，实现子维度级软依赖、prompt 自适应组装、调用编排与失败预算。

## 依赖

Spec 01（Prompt 模板就绪）、Spec 02（MockBackend 可用于测试）。

## 产出文件

- `video_eval/evaluators/vlm_judge.py`（新建）

## 类定义（详设 §6.1）

```python
@register_evaluator("vlm_judge")
class VLMJudge(BaseEvaluator):
    name = "vlm_judge"
    version = "0.1.0"
    device_requirement = "any"
    backend_config_key = "backend"
    requires = ["frames"]       # v7: only hard-dep frames; ASR/OCR are soft deps
    dimension_slots = {
        "main_image": ["sellpoint_coverage", "cross_modal"],
        "external": ["hook_strength", "marketing_logic", "audience_match", "cross_modal"],
        "general": ["cross_modal"],
    }
    default_weights = {
        "sellpoint_coverage": 0.20, "cross_modal": 0.15,
        "hook_strength": 0.20, "marketing_logic": 0.15, "audience_match": 0.10,
    }
    config_schema = {
        "backend": {"type": "str", "default": "mock", "required": True},
        "api_concurrency": {"type": "int", "default": 4},
        "api_max_failures": {"type": "int", "default": 5},
        "dimensions_main_image": {"type": "list", "default": ["sellpoint_coverage", "cross_modal"]},
        "dimensions_external": {"type": "list", "default": ["hook_strength", "marketing_logic", "audience_match", "cross_modal"]},
        "dimensions_general": {"type": "list", "default": ["cross_modal"]},
    }
```

## 核心实现

### __enter__（Backend 生命周期 §6.1）

```python
def __enter__(self):
    backend_name = self.config.get("backend", "mock")
    backends_config = self.config.get("_backends", {})
    backend_cfg = backends_config.get(backend_name, {})
    
    BackendCls = backend_registry.get(backend_name)
    self._backend = BackendCls(self.device_manager, backend_cfg)
    self._backend.__enter__()
    
    # Load prompt templates
    self._templates = self._load_templates()
    return self
```

### __exit__

```python
def __exit__(self, *args):
    if self._backend:
        self._backend.__exit__(*args)
        self._backend = None
```

### evaluate()（v7/A2 子维度软依赖）

```python
def evaluate(self, context) -> list[EvalResult]:
    dimensions = self.slots_for(context.video_type)
    
    # Soft dependency check
    asr_available = (context.asr is not None 
                     and "asr" not in context.extraction_failures)
    ocr_available = (bool(context.ocr)
                     and "ocr" not in context.extraction_failures)
    
    results = []
    failure_count = 0
    max_failures = self.config.get("api_max_failures", 5)
    
    for dim in dimensions:
        # Failure budget check
        if failure_count >= max_failures:
            results.append(self._placeholder(dim, reason="evaluation_failed"))
            continue
        
        # Sub-dimension dependency check
        if dim == "sellpoint_coverage" and context.product_info is None:
            results.append(self._placeholder(dim, reason="missing_product_info"))
            continue
        if dim == "cross_modal" and not asr_available and not ocr_available:
            results.append(self._placeholder(dim, reason="missing_dependency"))
            continue
        
        # Build prompt with adaptive modality sections
        prompt = self._build_prompt(dim, context, asr_available, ocr_available)
        
        # Call backend
        try:
            vlm_result = self._backend.judge(context, prompt)
            result = self._convert(dim, vlm_result, asr_available, ocr_available)
            results.append(result)
        except VLMOutputParseError:
            failure_count += 1
            results.append(self._placeholder(dim, reason="parse_failed"))
        except Exception:
            failure_count += 1
            results.append(self._placeholder(dim, reason="evaluation_failed"))
    
    return results
```

### _build_prompt()

```python
def _build_prompt(self, dim, context, asr_available, ocr_available) -> str:
    template = self._templates[dim]
    
    # Replace placeholders
    rendered = template
    rendered = rendered.replace("{video_type}", context.video_type)
    
    if dim == "sellpoint_coverage" and context.product_info:
        rendered = rendered.replace("{product_title}", context.product_info.title)
        rendered = rendered.replace("{selling_points}", 
                                    "\n".join(f"- {sp}" for sp in context.product_info.selling_points))
    
    # Conditional ASR section
    if asr_available and context.asr:
        rendered = rendered.replace("{asr_text}", context.asr.full_text)
        # Keep ASR section
    else:
        # Remove ASR section between markers
        rendered = self._remove_section(rendered, "asr_available")
    
    # Conditional OCR section  
    if ocr_available and context.ocr:
        ocr_text = "\n".join(item.text for item in context.ocr)
        rendered = rendered.replace("{ocr_text}", ocr_text)
    else:
        rendered = self._remove_section(rendered, "ocr_available")
    
    # Modality note
    modalities = ["visual"]
    if asr_available: modalities.append("audio/speech")
    if ocr_available: modalities.append("on-screen text")
    rendered = rendered.replace("{modality_note}", ", ".join(modalities))
    
    return rendered
```

### _convert()

```python
def _convert(self, dim, vlm_result, asr_available, ocr_available) -> EvalResult:
    evidence = vlm_result.evidence
    # Add input_modalities to evidence
    input_modalities = ["visual"]
    if asr_available: input_modalities.append("asr")
    if ocr_available: input_modalities.append("ocr")
    
    return EvalResult(
        dimension=dim,
        evaluator="vlm_judge",
        score=vlm_result.score,
        status="scored",
        evidence={"vlm_evidence": [e.model_dump() for e in evidence], 
                  "input_modalities": input_modalities},
        reasoning=vlm_result.reasoning,
        suggestion=vlm_result.suggestion,
    )
```

### _placeholder()

```python
def _placeholder(self, dim, reason) -> EvalResult:
    return EvalResult(
        dimension=dim,
        evaluator="vlm_judge",
        score=0.0,
        status="error" if reason in ("evaluation_failed", "parse_failed") else "skipped",
        reason=reason,
    )
```

## 验收标准

- [ ] `video-eval plugins` 列出 vlm_judge（requires=[frames], slots 正确）
- [ ] `video-eval eval --video x.mp4 --video-type external --set evaluators.vlm_judge.backend=mock`：输出 4 个子维度
- [ ] `video-eval eval --video x.mp4 --video-type general --set evaluators.vlm_judge.backend=mock`：输出 1 个子维度（cross_modal）
- [ ] ASR 不可用时：hook_strength/marketing_logic/audience_match 仍评分（evidence.input_modalities 无 "asr"）
- [ ] ASR+OCR 均不可用时：cross_modal → skipped: missing_dependency
- [ ] product_info 缺失时：sellpoint_coverage → skipped: missing_product_info
- [ ] api_max_failures=2, mock 模拟 3 次失败：第 3 个子维度直接占位
- [ ] Backend 在 __enter__ 启动、__exit__ 释放
