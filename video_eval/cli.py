"""CLI entry point for video-eval framework (Spec 10).

Connects Pipeline, ConfigLoader, Registry, and DeviceManager into
a user-facing command-line tool using Click.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from video_eval import __version__


# ---------------------------------------------------------------------------
# Exit code calculation
# ---------------------------------------------------------------------------


def _compute_exit_code(report: Any) -> int:
    """Compute exit code from a FinalReport.

    Priority: 6 > 5 > 4 > 3 > 1 > 0
    - 6: pipeline interrupted (ExtractionError, handled separately)
    - 5: no scored dimensions and no error dimensions
    - 4: any error dimension (regardless of grade)
    - 3: REJECT
    - 1: grade C
    - 0: grade A or B, no errors
    """
    has_error = any(
        r.status == "error" for r in report.dimension_results.values()
    )
    has_scored = any(
        r.status == "scored" for r in report.dimension_results.values()
    )

    if not has_scored and not has_error:
        return 5
    if has_error:
        return 4
    if report.grade == "REJECT":
        return 3
    if report.grade == "C":
        return 1
    return 0


def _compute_batch_exit_code(results: list) -> int:
    """Max severity across all batch items."""
    codes: list[int] = []
    for r in results:
        if r.error:
            codes.append(6)
        elif r.report:
            codes.append(_compute_exit_code(r.report))
    if not codes:
        return 0
    priority = {6: 6, 5: 5, 4: 4, 3: 3, 1: 2, 0: 1}
    return max(codes, key=lambda c: priority.get(c, 0))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity flag."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__)
def main():
    """video-eval: AI-generated video quality evaluation framework."""
    pass


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------


@main.command("eval")
@click.option("--video", required=True, type=click.Path(exists=True), help="Path to the video file.")
@click.option(
    "--video-type",
    required=True,
    type=click.Choice(["main_image", "external", "general"]),
    help="Video type category.",
)
@click.option("--product-title", default=None, help="Product title for main_image type.")
@click.option("--selling-points", multiple=True, help="Product selling points (repeatable).")
@click.option(
    "--product-images",
    multiple=True,
    type=click.Path(exists=True),
    help="Product main image paths (repeatable).",
)
@click.option("--config", "config_path", default=None, type=click.Path(), help="Config YAML path.")
@click.option("--output", default=None, type=click.Path(), help="Output file path (default: stdout).")
@click.option("--set", "overrides", multiple=True, help="Config override in KEY=VALUE format.")
@click.option(
    "--device",
    default=None,
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    help="Device preference.",
)
@click.option("--no-meta", is_flag=True, help="Exclude meta from output.")
@click.option("--no-evidence", is_flag=True, help="Exclude evidence from output.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def eval_cmd(
    video: str,
    video_type: str,
    product_title: str | None,
    selling_points: tuple[str, ...],
    product_images: tuple[str, ...],
    config_path: str | None,
    output: str | None,
    overrides: tuple[str, ...],
    device: str | None,
    no_meta: bool,
    no_evidence: bool,
    verbose: bool,
) -> None:
    """Evaluate a single video."""
    _setup_logging(verbose)

    from video_eval.core.config import ConfigLoader
    from video_eval.core.device import DeviceManager
    from video_eval.core.exceptions import ConfigError, ExtractionError
    from video_eval.core.pipeline import Pipeline
    from video_eval.core.registry import initialize_registries
    from video_eval.core.schemas import ProductInfo

    # 1. Initialize registries
    initialize_registries()

    # 2. Load config
    loader = ConfigLoader(config_path)
    try:
        config = loader.load()
        config = loader.merge_cli_overrides(config, list(overrides))
    except ConfigError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(2)

    # 3. Validate config
    issues = loader.validate(config)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e.message}", err=True)
        sys.exit(2)

    # 4. Setup device and pipeline
    preferred = None if device in (None, "auto") else device
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

    # 6. Run pipeline
    try:
        report = pipeline.run(video, product_info, video_type)
    except ConfigError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(2)
    except ExtractionError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(6)

    # 7. Format output
    output_dict = report.model_dump()
    if no_meta:
        output_dict.pop("meta", None)
    if no_evidence:
        for dr in output_dict.get("dimension_results", {}).values():
            if isinstance(dr, dict):
                dr.pop("evidence", None)

    pretty = config.get("output", {}).get("pretty", True)
    json_str = json.dumps(
        output_dict, ensure_ascii=False, indent=2 if pretty else None, default=str
    )

    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        click.echo(f"Report written to {output}")
    else:
        click.echo(json_str)

    # 8. Exit code
    sys.exit(_compute_exit_code(report))


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------


@main.command("batch")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input directory or manifest CSV.",
)
@click.option("--config", "config_path", default=None, type=click.Path(), help="Config YAML path.")
@click.option("--output", default=None, type=click.Path(), help="Output file path.")
@click.option(
    "--output-format",
    default="json",
    type=click.Choice(["json", "jsonl"]),
    help="Output format.",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    help="Device preference.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--fail-fast", is_flag=True, help="Stop on first error.")
@click.option("--set", "overrides", multiple=True, help="Config override in KEY=VALUE format.")
def batch_cmd(
    input_path: str,
    config_path: str | None,
    output: str | None,
    output_format: str,
    device: str | None,
    verbose: bool,
    fail_fast: bool,
    overrides: tuple[str, ...],
) -> None:
    """Batch evaluate videos."""
    _setup_logging(verbose)

    from video_eval.core.config import ConfigLoader
    from video_eval.core.device import DeviceManager
    from video_eval.core.exceptions import ConfigError
    from video_eval.core.pipeline import Pipeline
    from video_eval.core.registry import initialize_registries
    from video_eval.core.schemas import BatchItem, ProductInfo

    # 1. Initialize registries
    initialize_registries()

    # 2. Load config
    loader = ConfigLoader(config_path)
    try:
        config = loader.load()
        config = loader.merge_cli_overrides(config, list(overrides))
    except ConfigError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(2)

    # 3. Validate config
    issues = loader.validate(config)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e.message}", err=True)
        sys.exit(2)

    # 4. Setup
    preferred = None if device in (None, "auto") else device
    dm = DeviceManager(preferred)
    pipeline = Pipeline(config, dm)

    # 5. Build batch items from input
    items = _build_batch_items(input_path)
    if not items:
        click.echo("ERROR: No video files found in input.", err=True)
        sys.exit(2)

    click.echo(f"Processing {len(items)} video(s)...", err=True)

    # 6. Run batch
    if fail_fast:
        from video_eval.core.exceptions import ExtractionError
        from video_eval.core.schemas import BatchItemResult

        # Run one-by-one with early exit
        results = []
        for item in items:
            try:
                report = pipeline.run(item.video_path, item.product_info, item.video_type)
                results.append(BatchItemResult(item=item, report=report))
            except ExtractionError as exc:
                results.append(BatchItemResult(item=item, error=str(exc)))
                break
            except Exception as exc:
                results.append(BatchItemResult(item=item, error=f"Unexpected error: {exc}"))
                break
    else:
        results = pipeline.run_batch(items)

    # 7. Output results
    output_data = _format_batch_results(results, output_format)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(output_data, encoding="utf-8")
        click.echo(f"Results written to {output}", err=True)
    else:
        click.echo(output_data)

    # 8. Summary
    total = len(results)
    passed = sum(1 for r in results if r.report and r.report.passed)
    failed = sum(1 for r in results if r.error)
    click.echo(
        f"\nSummary: {total} total, {passed} passed, "
        f"{total - passed - failed} not passed, {failed} errors",
        err=True,
    )

    # 9. Exit code
    sys.exit(_compute_batch_exit_code(results))


def _build_batch_items(input_path: str) -> list:
    """Build BatchItem list from a directory or manifest CSV."""
    from video_eval.core.schemas import BatchItem, ProductInfo

    path = Path(input_path)
    items: list[BatchItem] = []

    if path.is_file() and path.suffix == ".csv":
        # CSV manifest: columns video_path, video_type, [product_title, selling_points, product_images]
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = row.get("video_path", "").strip()
                video_type = row.get("video_type", "general").strip()
                if not video_path:
                    continue

                product_info = None
                title = row.get("product_title", "").strip()
                if title:
                    sp = row.get("selling_points", "")
                    images = row.get("product_images", "")
                    product_info = ProductInfo(
                        title=title,
                        selling_points=[s.strip() for s in sp.split("|") if s.strip()],
                        main_image_paths=[s.strip() for s in images.split("|") if s.strip()],
                    )

                items.append(
                    BatchItem(
                        video_path=video_path,
                        video_type=video_type,
                        product_info=product_info,
                    )
                )
    elif path.is_dir():
        # Directory scan: find all video files, assume "general" type
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        for video_file in sorted(path.iterdir()):
            if video_file.suffix.lower() in video_exts and video_file.is_file():
                items.append(
                    BatchItem(
                        video_path=str(video_file),
                        video_type="general",
                    )
                )
    else:
        click.echo(f"ERROR: Input '{input_path}' is not a directory or CSV file.", err=True)

    return items


def _format_batch_results(results: list, output_format: str) -> str:
    """Format batch results as JSON or JSONL."""
    entries = []
    for r in results:
        entry: dict[str, Any] = {"video_path": r.item.video_path}
        if r.error:
            entry["error"] = r.error
        elif r.report:
            entry.update(r.report.model_dump())
        entries.append(entry)

    if output_format == "jsonl":
        return "\n".join(
            json.dumps(e, ensure_ascii=False, default=str) for e in entries
        )
    else:
        return json.dumps(entries, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# plugins command
# ---------------------------------------------------------------------------


@main.command("plugins")
@click.option("--detail", default=None, help="Show detail for a specific plugin.")
@click.option("--available-only", is_flag=True, help="Show only device-available plugins.")
@click.option(
    "--type",
    "plugin_type",
    default=None,
    type=click.Choice(["evaluator", "backend", "extractor", "fusion"]),
    help="Filter by plugin type.",
)
def plugins_cmd(
    detail: str | None,
    available_only: bool,
    plugin_type: str | None,
) -> None:
    """List registered plugins (fast path, no materialization)."""
    from video_eval.core.device import DeviceManager
    from video_eval.core.registry import (
        backend_registry,
        evaluator_registry,
        extractor_registry,
        fusion_registry,
        initialize_registries,
    )

    initialize_registries()

    dm = DeviceManager()

    registries = {
        "evaluator": evaluator_registry,
        "backend": backend_registry,
        "extractor": extractor_registry,
        "fusion": fusion_registry,
    }

    if plugin_type:
        registries = {plugin_type: registries[plugin_type]}

    if detail:
        # Show detail for a specific plugin
        _show_plugin_detail(detail, registries, dm)
        return

    # Table output
    for reg_name, registry in registries.items():
        metas = registry.list_meta()
        if not metas:
            continue

        click.echo(f"\n{'=' * 60}")
        click.echo(f"  {reg_name.upper()} PLUGINS ({len(metas)})")
        click.echo(f"{'=' * 60}")

        if reg_name == "extractor":
            # Show PROVIDES column for extractors
            header = f"{'NAME':<25} {'DEVICE':<8} {'PROVIDES':<30} {'ORIGIN':<12}"
            click.echo(header)
            click.echo("-" * 75)
            for meta in metas:
                if available_only and not dm.satisfies(meta.device_requirement):
                    continue
                provides_str = ", ".join(meta.provides) if meta.provides else "-"
                click.echo(
                    f"{meta.name:<25} {meta.device_requirement:<8} "
                    f"{provides_str:<30} {meta.origin:<12}"
                )
        elif reg_name == "evaluator":
            header = f"{'NAME':<25} {'DEVICE':<8} {'REQUIRES':<25} {'ORIGIN':<12}"
            click.echo(header)
            click.echo("-" * 70)
            for meta in metas:
                if available_only and not dm.satisfies(meta.device_requirement):
                    continue
                requires_str = ", ".join(meta.requires) if meta.requires else "-"
                click.echo(
                    f"{meta.name:<25} {meta.device_requirement:<8} "
                    f"{requires_str:<25} {meta.origin:<12}"
                )
        else:
            header = f"{'NAME':<25} {'DEVICE':<8} {'ORIGIN':<12}"
            click.echo(header)
            click.echo("-" * 45)
            for meta in metas:
                if available_only and not dm.satisfies(meta.device_requirement):
                    continue
                click.echo(
                    f"{meta.name:<25} {meta.device_requirement:<8} {meta.origin:<12}"
                )

    click.echo()


def _show_plugin_detail(name: str, registries: dict, dm: Any) -> None:
    """Show detailed info for a single plugin."""
    for reg_name, registry in registries.items():
        if registry.has(name):
            meta = registry.get_meta(name)
            click.echo(f"\nPlugin: {meta.name}")
            click.echo(f"  Type:        {reg_name}")
            click.echo(f"  Version:     {meta.version}")
            click.echo(f"  Device:      {meta.device_requirement}")
            click.echo(f"  Available:   {dm.satisfies(meta.device_requirement)}")
            click.echo(f"  Origin:      {meta.origin}")
            click.echo(f"  Placeholder: {meta.is_placeholder}")
            if meta.requires:
                click.echo(f"  Requires:    {', '.join(meta.requires)}")
            if meta.provides:
                click.echo(f"  Provides:    {', '.join(meta.provides)}")
            if meta.criticality != "optional":
                click.echo(f"  Criticality: {meta.criticality}")
            if meta.dimension_slots:
                click.echo(f"  Slots:       {json.dumps(meta.dimension_slots)}")
            if meta.config_schema:
                click.echo(f"  Config:")
                for key, spec in meta.config_schema.items():
                    click.echo(f"    {key}: {spec}")
            click.echo()
            return

    click.echo(f"Plugin '{name}' not found in any registry.", err=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# config group
# ---------------------------------------------------------------------------


@main.group("config")
def config_group():
    """Configuration management."""
    pass


@config_group.command("check")
@click.option("--config", "config_path", default=None, type=click.Path(), help="Config YAML path.")
@click.option("--set", "overrides", multiple=True, help="Config override in KEY=VALUE format.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def config_check(config_path: str | None, overrides: tuple[str, ...], verbose: bool) -> None:
    """Validate configuration."""
    _setup_logging(verbose)

    from video_eval.core.config import ConfigLoader
    from video_eval.core.exceptions import ConfigError
    from video_eval.core.registry import initialize_registries

    initialize_registries()

    loader = ConfigLoader(config_path)
    try:
        config = loader.load()
        config = loader.merge_cli_overrides(config, list(overrides))
    except ConfigError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(2)

    issues = loader.validate(config)

    if not issues:
        click.echo("Configuration valid.")
        sys.exit(0)

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if warnings:
        click.echo(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            plugin_str = f" [{w.plugin_name}]" if w.plugin_name else ""
            click.echo(f"  {w.message}{plugin_str}")

    if errors:
        click.echo(f"\nErrors ({len(errors)}):")
        for e in errors:
            plugin_str = f" [{e.plugin_name}]" if e.plugin_name else ""
            click.echo(f"  {e.message}{plugin_str}")

    click.echo()
    if errors:
        click.echo(f"Result: INVALID ({len(errors)} error(s), {len(warnings)} warning(s))")
        sys.exit(2)
    else:
        click.echo(f"Result: VALID ({len(warnings)} warning(s))")
        sys.exit(0)


# ---------------------------------------------------------------------------
# device group
# ---------------------------------------------------------------------------


@main.group("device")
def device_group():
    """Device information."""
    pass


@device_group.command("info")
def device_info() -> None:
    """Show device information."""
    from video_eval.core.device import DeviceManager

    dm = DeviceManager()
    mem = dm.memory_info()

    click.echo("\nDevice Information")
    click.echo("=" * 40)
    click.echo(f"  Device type:   {dm.device_type}")
    click.echo(f"  Is GPU:        {dm.is_gpu()}")
    click.echo(f"  Total memory:  {mem['total_gb']:.2f} GB")
    click.echo(f"  Free memory:   {mem['free_gb']:.2f} GB")

    # Torch availability
    try:
        import torch

        click.echo(f"  PyTorch:       {torch.__version__}")
        if dm.device_type == "cuda":
            click.echo(f"  CUDA version:  {torch.version.cuda}")
            click.echo(f"  GPU name:      {torch.cuda.get_device_name(0)}")
    except ImportError:
        click.echo("  PyTorch:       not installed")

    # Backend availability hints
    click.echo("\nBackend Status")
    click.echo("-" * 40)

    try:
        import faster_whisper  # noqa: F401

        click.echo("  faster-whisper: available")
    except ImportError:
        click.echo("  faster-whisper: not installed")

    try:
        import open_clip  # noqa: F401

        click.echo("  open-clip:      available")
    except ImportError:
        click.echo("  open-clip:      not installed")

    click.echo()
