# Phase 3 Spec 02：MockBackend 增强

## 目标

增强 MockBackend 以支持按维度返回不同分数、模拟延迟、输出标准 VLMResult 格式。用于测试 VLMJudge 的全流程而无需真实 VLM。

## 修改文件

- `video_eval/backends/mock.py`（增强已有实现）

## 增强内容

### 1. 按维度返回不同分数

```python
config_schema = {
    "default_score": {"type": "float", "default": 0.75},
    "dimension_scores": {"type": "dict", "default": {}},
    "delay": {"type": "float", "default": 0.0},
}
```

配置示例：
```yaml
backends:
  mock:
    default_score: 0.75
    dimension_scores:
      compliance: 1.0
      cross_modal: 0.5
      hook_strength: 0.8
    delay: 0.0  # seconds, for timeout testing
```

### 2. judge() 增强

```python
def judge(self, context, prompt: str) -> VLMResult:
    # Extract dimension name from prompt (pattern match)
    dimension = self._extract_dimension_from_prompt(prompt)
    score = self._dimension_scores.get(dimension, self._default_score)
    
    # Simulate delay
    if self._delay > 0:
        import time
        time.sleep(self._delay)
    
    # Map to 5-level (reverse of score=(level-1)/4)
    level = round(score * 4) + 1
    level = max(1, min(5, level))
    
    return VLMResult(
        score=score,
        reasoning=f"Mock evaluation for dimension '{dimension}': level {level}/5",
        evidence=[EvidenceItem(modality="visual", timestamp=0.0, detail=f"Mock: scored {score:.2f}")],
        suggestion=f"Mock suggestion for {dimension}" if score < 0.75 else "",
        raw_output=json.dumps({"level": level, "reasoning": "mock", "evidence": [], "suggestion": ""}),
    )
```

### 3. 从 prompt 提取维度名

```python
def _extract_dimension_from_prompt(self, prompt: str) -> str:
    """Extract dimension name from rendered prompt text."""
    # Look for dimension markers in prompt
    for dim in ("sellpoint_coverage", "cross_modal", "hook_strength", 
                "marketing_logic", "audience_match"):
        if dim in prompt.lower() or dim.replace("_", " ") in prompt.lower():
            return dim
    return "unknown"
```

## 验收标准

- [ ] `MockBackend(dm, {"default_score": 0.5}).judge(ctx, "cross_modal")` 返回 score=0.5
- [ ] `MockBackend(dm, {"dimension_scores": {"cross_modal": 0.25}}).judge(ctx, "...cross_modal...")` 返回 score=0.25
- [ ] VLMResult 包含完整字段（reasoning/evidence/suggestion/raw_output）
- [ ] delay 配置生效（可观测延迟）
- [ ] 输出格式与真实 backend 一致（可被 VLMJudge 消费）
