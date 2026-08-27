# Spec 07：融合决策

## 目标

实现 `WeightedVetoFusion` 策略，完成否决扫描 → 权重计算 → 等级判定 → 建议生成的完整融合流程。

## 依赖

Spec 02（EvalResult, FusionOutcome）、Spec 05（BaseFusion）。

## 产出文件

- `video_eval/fusions/__init__.py`
- `video_eval/fusions/weighted_veto.py`

## WeightedVetoFusion 实现

参考详设 §5.1 完整伪代码。

```python
from video_eval.core.base import BaseFusion
from video_eval.core.registry import register_fusion
from video_eval.core.schemas import EvalResult, FusionOutcome

@register_fusion("weighted_veto")
class WeightedVetoFusion(BaseFusion):
    name = "weighted_veto"
    version = "0.1.0"

    # Reasons that trigger strict_veto rejection (D5)
    STRICT_VETO_TRIGGERS = {"init_failed", "evaluation_failed", "parse_failed", "runtime_unavailable"}

    def __init__(self, config: dict):
        super().__init__(config)
        self.thresholds = config.get("thresholds", {"A": 0.75, "B": 0.60, "C": 0.40})
        self.veto_thresholds = config.get("veto_thresholds", {})
        self.strict_veto_dims = set(config.get("strict_veto_dims", []))

    def fuse(
        self,
        results: dict[str, EvalResult],
        video_type: str,
        weights: dict,
        default_weights: dict[str, float | None],
    ) -> FusionOutcome:
        ...
```

### fuse() 完整流程

```
Step 1: Veto scan
    veto_dims = set(veto_thresholds.keys()) | strict_veto_dims
    rejected = False
    veto_reasons = []

    for name in veto_dims:
        result = results.get(name)
        if result is None:
            continue
        if result.status != "scored":
            if name in strict_veto_dims and result.reason in STRICT_VETO_TRIGGERS:
                rejected = True
                veto_reasons.append(f"{name} execution failed ({result.reason}), strict veto dimension has no result")
            elif name in strict_veto_dims:
                # Environment issue not caught by pre-check (should not be reachable)
                log.warning(...)
            continue
        threshold = veto_thresholds.get(name, 0.0)
        if result.score <= threshold:
            rejected = True
            veto_reasons.append(f"{name} score {result.score:.2f} did not exceed veto threshold {threshold}")

Step 2: Weight calculation
    scored_dims = [d for d, r in results.items() if r.status == "scored"]
    veto_only_dims = {d for d in veto_dims
                      if weights.get(d) is None and default_weights.get(d) is None}
    weighted_dims = [d for d in scored_dims if d not in veto_only_dims]

    if rejected:
        overall_score = 0.0
    elif not weighted_dims:
        overall_score = 0.0
        # Will become REJECT below
    else:
        raw_weights = {}
        for d in weighted_dims:
            w = weights.get(d)
            if w is None:
                w = default_weights.get(d)
            if w is None:
                log.warning(f"Dimension {d} has no weight (neither config nor default_weights), treating as 0")
                w = 0.0
            raw_weights[d] = w
        total = sum(raw_weights.values())
        if total == 0:
            overall_score = 0.0
        else:
            normalized = {d: w/total for d, w in raw_weights.items()}
            overall_score = sum(normalized[d] * results[d].score for d in weighted_dims)

Step 3: Grade determination
    if rejected:
        grade = "REJECT"
    elif overall_score >= thresholds["A"]:
        grade = "A"
    elif overall_score >= thresholds["B"]:
        grade = "B"
    elif overall_score >= thresholds["C"]:
        grade = "C"
    else:
        grade = "REJECT"

Step 4: Suggestions
    suggestion_threshold = thresholds["B"]
    suggestions = [
        f"[{d}] {results[d].suggestion}"
        for d in scored_dims
        if results[d].score < suggestion_threshold and results[d].suggestion
    ]

Step 5: Return
    return FusionOutcome(
        overall_score=overall_score,
        grade=grade,
        passed=(grade in ("A", "B")),
        veto_reasons=veto_reasons,
        suggestions=suggestions,
    )
```

## 验收标准

- [ ] 全维度 scored=1.0 → grade="A", passed=True, veto_reasons=[]
- [ ] compliance score=0.0, veto_threshold=0.0 → rejected, grade="REJECT"
- [ ] compliance `status="error", reason="evaluation_failed"` + strict_veto → rejected
- [ ] compliance `status="skipped", reason="extraction_failed"` + strict_veto → **NOT** rejected（不在 STRICT_VETO_TRIGGERS）
- [ ] product_fidelity 同时在 veto_thresholds 和 weights → 既参与否决也参与加权
- [ ] compliance 只在 veto_thresholds 不在 weights 且 default_weights=None → veto_only，不参与加权
- [ ] 全部维度 skipped → overall_score=0.0, grade="REJECT"
- [ ] weights 和不为 1 时自动归一化
- [ ] default_weights 兜底：weights 无该维度但 default_weights 有 → 正确使用
- [ ] suggestions 只收集 score < thresholds.B 的维度
