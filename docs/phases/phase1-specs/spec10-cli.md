# Spec 10：CLI 实现

## 目标

将 Spec 01 的 CLI 骨架填充为完整实现，对接 Pipeline/Config/Registry，处理退出码与输出格式。

## 依赖

Spec 06（ConfigLoader）、Spec 09（Pipeline）、Spec 04（Registry 初始化）。

## 产出文件

- `video_eval/cli.py`（替换 spec01 的占位）

## 命令实现

### eval 命令

```python
@main.command()
@click.option("--video", required=True, type=click.Path(exists=True))
@click.option("--video-type", required=True, type=click.Choice(["main_image", "external", "general"]))
@click.option("--product-title", default=None)
@click.option("--selling-points", multiple=True)
@click.option("--product-images", multiple=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default=None, type=click.Path())
@click.option("--output", default=None, type=click.Path())
@click.option("--set", "overrides", multiple=True)
@click.option("--device", default=None, type=click.Choice(["auto", "cuda", "mps", "cpu"]))
@click.option("--no-meta", is_flag=True)
@click.option("--no-evidence", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
def eval_cmd(video, video_type, product_title, selling_points, product_images,
             config_path, output, overrides, device, no_meta, no_evidence, verbose):
    """Evaluate a single video."""
    # 1. Initialize registries
    initialize_registries()

    # 2. Load config
    loader = ConfigLoader(config_path)
    config = loader.load()
    config = loader.merge_cli_overrides(config, list(overrides))

    # 3. Validate config
    issues = loader.validate(config)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e.message}", err=True)
        sys.exit(2)

    # 4. Setup
    preferred = None if device == "auto" else device
    dm = DeviceManager(preferred)
    pipeline = Pipeline(config, dm)

    # 5. Build ProductInfo
    product_info = None
    if product_title:
        product_info = ProductInfo(
            title=product_title,
            selling_points=list(selling_points),
            main_image_paths=list(product_images),
        )

    # 6. Run
    try:
        report = pipeline.run(video, product_info, video_type)
    except ConfigError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(2)
    except ExtractionError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(6)

    # 7. Output
    output_dict = report.model_dump()
    if no_meta:
        output_dict.pop("meta", None)
    if no_evidence:
        for dr in output_dict.get("dimension_results", {}).values():
            dr.pop("evidence", None)

    json_str = json.dumps(output_dict, ensure_ascii=False, indent=2 if config.get("output", {}).get("pretty", True) else None)
    if output:
        Path(output).write_text(json_str)
    else:
        click.echo(json_str)

    # 8. Exit code
    sys.exit(_compute_exit_code(report))
```

### batch 命令

```python
@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default=None)
@click.option("--output", default=None, type=click.Path())
@click.option("--output-format", default="json", type=click.Choice(["json", "jsonl"]))
@click.option("--device", default=None)
@click.option("-v", "--verbose", is_flag=True)
@click.option("--fail-fast", is_flag=True)
@click.option("--set", "overrides", multiple=True)
def batch_cmd(...):
    """Batch evaluate videos."""
    # Similar setup...
    # Build items from input (directory scan or manifest.csv)
    # Call pipeline.run_batch(items)
    # Output results
    # Exit code = max severity across all items
    ...
```

### plugins 命令

```python
@main.command()
@click.option("--detail", default=None)
@click.option("--available-only", is_flag=True)
@click.option("--type", "plugin_type", default=None, type=click.Choice(["evaluator", "backend", "extractor", "fusion"]))
def plugins_cmd(detail, available_only, plugin_type):
    """List registered plugins."""
    initialize_registries()  # NO materialization for fast path
    # Format table output per §8.4
    ...
```

### config check 命令

```python
@config.command("check")
@click.option("--config", "config_path", default=None)
@click.option("--set", "overrides", multiple=True)
def config_check(config_path, overrides):
    """Validate configuration."""
    initialize_registries()
    loader = ConfigLoader(config_path)
    config = loader.load()
    config = loader.merge_cli_overrides(config, list(overrides))
    issues = loader.validate(config)
    # Pretty-print per §8.5 format
    ...
    sys.exit(2 if any(i.severity == "error" for i in issues) else 0)
```

### device info 命令

```python
@device.command("info")
def device_info():
    """Show device information."""
    dm = DeviceManager()
    mem = dm.memory_info()
    # Format per §8.5
    ...
```

## 退出码计算

```python
def _compute_exit_code(report: FinalReport) -> int:
    """
    Priority: 6 > 5 > 4 > 3 > 1 > 0
    - 2: pre-flight (handled separately, never reaches here)
    - 6: pipeline interrupted (ExtractionError, handled separately)
    - 5: no scored dimensions and no error dimensions
    - 4: any error dimension (regardless of grade)
    - 3: REJECT
    - 1: grade C
    - 0: grade A or B, no errors
    """
    has_error = any(r.status == "error" for r in report.dimension_results.values())
    has_scored = any(r.status == "scored" for r in report.dimension_results.values())

    if not has_scored and not has_error:
        return 5
    if has_error:
        return 4
    if report.grade == "REJECT":
        return 3
    if report.grade == "C":
        return 1
    return 0


def _compute_batch_exit_code(results: list[BatchItemResult]) -> int:
    """Max severity across all items. 2 handled separately (pre-flight)."""
    codes = []
    for r in results:
        if r.error:
            codes.append(6)
        elif r.report:
            codes.append(_compute_exit_code(r.report))
    if not codes:
        return 0
    # Priority ordering
    priority = {6: 6, 5: 5, 4: 4, 3: 3, 1: 2, 0: 1}
    return max(codes, key=lambda c: priority.get(c, 0))
```

## 验收标准

- [ ] `video-eval eval --video test.mp4 --video-type general` → JSON stdout + 正确退出码
- [ ] `video-eval eval --video nonexist.mp4 ...` → 退出码 6（required extractor fail）
- [ ] `video-eval batch --input ./videos/ --output ./results/` → 批量输出
- [ ] `video-eval plugins` → 表格输出（< 1s）
- [ ] `video-eval plugins --type extractor` → PROVIDES 列
- [ ] `video-eval config check` → ✅/⚠️/❌ 格式化输出
- [ ] `video-eval device info` → 设备 + 内存 + 后端状态
- [ ] `--set evaluators.compliance.limit_words=最,第一` → 正确覆盖
- [ ] `--no-meta` → 输出无 meta 字段
- [ ] `--no-evidence` → 输出无 evidence 字段
- [ ] grade=A 无 error → 退出码 0
- [ ] grade=REJECT → 退出码 3
- [ ] 有 error 维度 → 退出码 4
- [ ] 全部 skipped 无 error → 退出码 5
