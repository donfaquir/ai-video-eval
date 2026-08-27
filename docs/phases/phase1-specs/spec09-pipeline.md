# Spec 09：Pipeline 编排

## 目标

实现 `Pipeline` 类的 `run()` 和 `run_batch()` 方法，完成从过滤到融合的完整编排流程。这是框架的核心调度器。

## 依赖

Spec 02-08（全部前置 spec）。

## 产出文件

- `video_eval/core/pipeline.py`

## Pipeline 类

参考详设 §4 全节。

```python
class Pipeline:
    def __init__(self, config: dict, device_manager: DeviceManager):
        self.config = config
        self.device_manager = device_manager

    def run(self, video_path: str, product_info: ProductInfo | None, video_type: str) -> FinalReport:
        """Single video evaluation. Full 13-step flow."""
        ...

    def run_batch(self, items: list[BatchItem]) -> list[BatchItemResult]:
        """Batch evaluation. Resident mode (Phase 1); sequential deferred to Phase 2."""
        ...
```

## run() 完整 13 步

```python
def run(self, video_path, product_info, video_type) -> FinalReport:
    # 1. Materialize all entry-point plugins
    self._materialize_entry_points()

    # 2. Pre-flight checks (D5/D9)
    self._preflight_checks(video_type)

    # 3. Discover evaluators
    evaluators = self._discover_evaluators(video_type)

    # 4. Compute available_fields
    available_fields = self._compute_available_fields(product_info)

    # 5. Filter evaluators (F1/F2/F3)
    filtered = self._filter_evaluators(evaluators, available_fields)

    # 6. Run extractors (closure + topo sort + field-level degradation)
    context = self._run_extractors(video_path, product_info, video_type, filtered)

    # 7. Re-filter after extraction
    filtered = self._refilter_after_extraction(filtered, context)

    # 8. Run evaluators (serial, with context manager)
    results = self._run_evaluators(context, filtered)

    # 9. Fill placeholders
    results = self._fill_placeholders(results, evaluators, filtered, video_type)

    # 10. Build default_weights
    default_weights = self._build_default_weights(evaluators)

    # 11. Fuse
    outcome = self._fuse(results, video_type, default_weights)

    # 12. Assemble report
    report = self._assemble_report(outcome, video_path, video_type, results, evaluators)

    # 13. Return
    return report
```

## 关键内部方法

### _materialize_entry_points()

物化全部四类 entry-point 插件。物化失败按类型处理（详设 §1.6）。

### _preflight_checks(video_type)

详设 §5.3 五行校验表：
1. strict_veto_dims 维度已注册/已启用/在 effective_slots 中
2. strict_veto_dims 维度未 broken
3. 依赖闭包可达
4. D9 probe（__enter__/__exit__ 实测）
5. backend 设备要求满足

任一不满足 → raise ConfigError (exit code 2)。

### _discover_evaluators(video_type)

```python
def _discover_evaluators(self, video_type: str) -> list[EvaluatorInfo]:
    """Scan registry metadata, compute effective_slots."""
    infos = []
    for meta in evaluator_registry.list_meta():
        config_section = self.config.get("evaluators", {}).get(meta.name, {})
        # Compute effective_slots (§1.3)
        if meta.dimension_slots is None:
            effective_slots = [meta.name]
        else:
            override_key = f"dimensions_{video_type}"
            override = config_section.get(override_key)
            effective_slots = override if override is not None else meta.dimension_slots.get(video_type, [])
        infos.append(EvaluatorInfo(meta=meta, config=config_section, effective_slots=effective_slots))
    return infos
```

### _compute_available_fields(product_info)

双层定义：
- 静态层：设备满足且未禁用的抽取器 provides 并集
- 运行时层：video_path/video_type 恒可用；product_info 仅当非 None

### _filter_evaluators(evaluators, available_fields)

三重过滤（F1/F2/F3），详设 §4.3。返回通过的子集；未通过的标记 status/reason。

### _run_extractors(...)

详设 §4.5 全流程：
1. required_keys = union of filtered evaluators' requires
2. Subtract base fields
3. Select extractor candidates (device + enabled)
4. Seed = provides ∩ required_keys
5. Closure expansion (expand())
6. Topo sort
7. Serial execution with context manager
8. Failures: required → ExtractionError; optional → extraction_failures

### _refilter_after_extraction(filtered, context)

详设 §4.5 `_refilter_after_extraction` 伪代码。

### _run_evaluators(context, filtered)

详设 §4.4 执行序列（6 步）：
- 逐个 evaluator: get → instantiate → __enter__ → check_availability → evaluate → __exit__
- 失败按步骤标记不同 reason

### _fill_placeholders(results, evaluators, filtered, video_type)

详设 §4.4 占位生成规则：
- 被过滤/跳过/error 的评估器按 effective_slots 展开
- 多槽位评估器展开所有子维度

### _build_default_weights(evaluators)

详设 §2.4 `_build_default_weights` 伪代码。

### _fuse(results, video_type, default_weights)

```python
def _fuse(self, results, video_type, default_weights):
    strategy_name = self.config.get("fusion", {}).get("strategy", "weighted_veto")
    fusion_cls = fusion_registry.get(strategy_name)
    fusion = fusion_cls(self.config.get("fusion", {}))
    weights = self.config.get("fusion", {}).get(f"weights_{video_type}", {})
    return fusion.fuse(results, video_type, weights, default_weights)
```

### _assemble_report(outcome, ...)

组装 FinalReport + ReportMeta（framework_version, device, backend, evaluator_versions, skipped, config_hash, timestamp）。

## run_batch() 实现（resident 模式）

```python
def run_batch(self, items: list[BatchItem]) -> list[BatchItemResult]:
    self._materialize_entry_points()
    self._preflight_checks_batch(items)

    # Per-item filter
    per_item_filtered = {i: self._filter_for_item(item) for i, item in enumerate(items)}

    # Union sets
    evaluators_to_load = union(per_item_filtered)
    extractors_to_load = self._compute_extractors_union(per_item_filtered)

    # Enter all
    extractor_instances = self._enter_extractors(extractors_to_load)
    evaluator_instances = self._enter_evaluators(evaluators_to_load)

    results = []
    for i, item in enumerate(items):
        try:
            context = self._run_extractors_resident(item, extractor_instances)
            item_filtered = self._refilter_after_extraction(per_item_filtered[i], context)
            eval_results = self._run_evaluators_resident(context, item_filtered, evaluator_instances)
            eval_results = self._fill_placeholders(...)
            default_weights = self._build_default_weights(...)
            outcome = self._fuse(eval_results, item.video_type, default_weights)
            report = self._assemble_report(outcome, item.video_path, item.video_type, ...)
            results.append(BatchItemResult(item=item, report=report))
        except ExtractionError as e:
            results.append(BatchItemResult(item=item, error=str(e)))

    # Exit all
    self._exit_all(extractor_instances + evaluator_instances)
    return results
```

## 验收标准

- [ ] `Pipeline(config, dm).run("test.mp4", None, "general")` 返回 FinalReport
- [ ] F1 过滤：disabled 评估器不执行、不生成占位
- [ ] F2 过滤：device 不满足的评估器生成 `skipped: device_unavailable` 占位
- [ ] F3 过滤：requires 缺失的评估器生成 `skipped: missing_dependency` 占位
- [ ] 抽取器闭包扩展：compliance requires ["asr","ocr"] → video_meta 被纳入（为 ocr 提供 frames）
- [ ] optional 抽取器失败 → 字段级降级 → 依赖评估器 `skipped: extraction_failed`
- [ ] required 抽取器失败 → ExtractionError（退出码 6）
- [ ] 多槽位占位展开：vlm_judge 跳过时按 effective_slots 展开全部子维度
- [ ] run_batch resident 模式：evaluator 只 enter 一次
- [ ] 前置校验失败 → ConfigError（退出码 2）
