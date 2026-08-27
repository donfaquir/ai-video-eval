"""Evaluation pipeline orchestration (Spec 09).

Implements the 13-step single-video flow (run) and resident-mode batch (run_batch).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from video_eval.core.config import compute_config_hash
from video_eval.core.device import DeviceManager
from video_eval.core.exceptions import ConfigError, ExtractionError, MaterializationError
from video_eval.core.registry import (
    Placeholder,
    backend_registry,
    evaluator_registry,
    extractor_registry,
    fusion_registry,
)
from video_eval.core.schemas import (
    BatchItem,
    BatchItemResult,
    EvalContext,
    EvalResult,
    EvaluatorInfo,
    FinalReport,
    FusionOutcome,
    PluginMeta,
    ProductInfo,
    ReportMeta,
)

logger = logging.getLogger(__name__)

__version__ = "0.1.0"


class Pipeline:
    """Core orchestration engine for video evaluation."""

    def __init__(self, config: dict, device_manager: DeviceManager) -> None:
        self.config = config
        self.device_manager = device_manager

    # ==================================================================
    # Public API
    # ==================================================================

    def run(
        self,
        video_path: str,
        product_info: ProductInfo | None,
        video_type: str,
    ) -> FinalReport:
        """Single video evaluation. Full 13-step flow."""
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
        report = self._assemble_report(
            outcome, video_path, video_type, results, evaluators
        )

        # 13. Return
        return report

    def run_batch(self, items: list[BatchItem]) -> list[BatchItemResult]:
        """Batch evaluation in resident mode.

        Materialize + preflight once; enter extractors/evaluators once;
        loop items; exit all at the end.
        """
        # 1. Materialize + preflight (use first item's video_type for preflight)
        self._materialize_entry_points()

        # Collect unique video_types for preflight
        video_types = {item.video_type for item in items}
        for vt in video_types:
            self._preflight_checks(vt)

        # 2. Per-item discover + filter
        per_item_evaluators: list[list[EvaluatorInfo]] = []
        for item in items:
            evaluators = self._discover_evaluators(item.video_type)
            available = self._compute_available_fields(item.product_info)
            filtered = self._filter_evaluators(evaluators, available)
            per_item_evaluators.append(evaluators)

        # 3. Loop items (in resident mode, we still process serially per item
        #    but use the single-item flow for simplicity in Phase 1)
        batch_results: list[BatchItemResult] = []
        for i, item in enumerate(items):
            try:
                report = self.run(item.video_path, item.product_info, item.video_type)
                batch_results.append(BatchItemResult(item=item, report=report))
            except ExtractionError as exc:
                batch_results.append(BatchItemResult(item=item, error=str(exc)))
            except Exception as exc:
                batch_results.append(
                    BatchItemResult(item=item, error=f"Unexpected error: {exc}")
                )

        return batch_results

    # ==================================================================
    # Step 1: Materialize entry points
    # ==================================================================

    def _materialize_entry_points(self) -> None:
        """Materialize all 4 registry entry-point plugins.

        Failures:
        - evaluators: mark broken (non-fatal, will be skipped later)
        - extractors: mark broken (check criticality at extraction time)
        - backends: ConfigError if the selected backend fails
        - fusions: ConfigError if the selected strategy fails
        """
        # Evaluators: attempt materialization, failures are non-fatal
        for name in evaluator_registry.list():
            if evaluator_registry.is_placeholder(name):
                try:
                    evaluator_registry.get(name)
                except MaterializationError:
                    logger.warning("Evaluator '%s' failed to materialize.", name)

        # Extractors: attempt materialization, failures are non-fatal here
        for name in extractor_registry.list():
            if extractor_registry.is_placeholder(name):
                try:
                    extractor_registry.get(name)
                except MaterializationError:
                    logger.warning("Extractor '%s' failed to materialize.", name)

        # Backend: the selected backend must succeed
        evaluators_cfg = self.config.get("evaluators", {})
        needed_backends: set[str] = set()
        for eval_name, eval_conf in evaluators_cfg.items():
            if not isinstance(eval_conf, dict):
                continue
            if not eval_conf.get("enabled", True):
                continue
            backend_ref = eval_conf.get("backend")
            if backend_ref:
                needed_backends.add(backend_ref)

        for backend_name in needed_backends:
            if backend_registry.has(backend_name) and backend_registry.is_placeholder(
                backend_name
            ):
                try:
                    backend_registry.get(backend_name)
                except MaterializationError as exc:
                    raise ConfigError(
                        f"Backend '{backend_name}' failed to materialize: {exc}"
                    ) from exc

        # Fusion: the selected strategy must succeed
        strategy_name = self.config.get("fusion", {}).get("strategy", "weighted_veto")
        if fusion_registry.has(strategy_name) and fusion_registry.is_placeholder(
            strategy_name
        ):
            try:
                fusion_registry.get(strategy_name)
            except MaterializationError as exc:
                raise ConfigError(
                    f"Fusion strategy '{strategy_name}' failed to materialize: {exc}"
                ) from exc

    # ==================================================================
    # Step 2: Pre-flight checks
    # ==================================================================

    def _preflight_checks(self, video_type: str) -> None:
        """Validate strict_veto_dims requirements before execution.

        Checks:
        1. strict_veto_dims dimensions are registered, enabled, and in effective_slots
        2. strict_veto_dims dimensions are not broken
        3. Dependency closure is reachable (simplified)
        4. D9 probe (simplified: check extractors not broken)
        5. Backend device requirements are satisfied
        """
        fusion_cfg = self.config.get("fusion", {})
        strict_veto_dims = fusion_cfg.get("strict_veto_dims", [])

        if not strict_veto_dims:
            return

        evaluators_cfg = self.config.get("evaluators", {})

        for dim in strict_veto_dims:
            # Find the evaluator that produces this dimension
            producer_found = False
            for meta in evaluator_registry.list_meta():
                eval_conf = evaluators_cfg.get(meta.name, {})
                if not isinstance(eval_conf, dict):
                    eval_conf = {}

                # Check enabled (F1)
                if not eval_conf.get("enabled", True):
                    continue

                # Compute effective_slots
                if meta.dimension_slots is None:
                    effective_slots = [meta.name]
                else:
                    override_key = f"dimensions_{video_type}"
                    override = eval_conf.get(override_key)
                    effective_slots = (
                        override
                        if override is not None
                        else meta.dimension_slots.get(video_type, [])
                    )

                if dim in effective_slots:
                    producer_found = True

                    # Check not broken
                    if evaluator_registry.is_broken(meta.name):
                        raise ConfigError(
                            f"strict_veto_dims dimension '{dim}' producer "
                            f"'{meta.name}' is broken (failed to materialize)"
                        )

                    # Check device requirement (F2)
                    if not self.device_manager.satisfies(meta.device_requirement):
                        raise ConfigError(
                            f"strict_veto_dims dimension '{dim}' producer "
                            f"'{meta.name}' requires device "
                            f"'{meta.device_requirement}' which is not available"
                        )

                    # Simplified D9: check required extractors are not broken
                    for req_field in meta.requires:
                        for ext_meta in extractor_registry.list_meta():
                            if req_field in ext_meta.provides:
                                if extractor_registry.is_broken(ext_meta.name):
                                    raise ConfigError(
                                        f"strict_veto_dims dimension '{dim}': "
                                        f"required extractor '{ext_meta.name}' "
                                        f"is broken"
                                    )
                    break

            if not producer_found:
                raise ConfigError(
                    f"strict_veto_dims dimension '{dim}' has no enabled "
                    f"producer evaluator for video_type '{video_type}'"
                )

    # ==================================================================
    # Step 3: Discover evaluators
    # ==================================================================

    def _discover_evaluators(self, video_type: str) -> list[EvaluatorInfo]:
        """Scan registry metadata, compute effective_slots."""
        infos: list[EvaluatorInfo] = []
        for meta in evaluator_registry.list_meta():
            config_section = self.config.get("evaluators", {}).get(meta.name, {})
            if not isinstance(config_section, dict):
                config_section = {}

            # Compute effective_slots
            if meta.dimension_slots is None:
                effective_slots = [meta.name]
            else:
                override_key = f"dimensions_{video_type}"
                override = config_section.get(override_key)
                effective_slots = (
                    override
                    if override is not None
                    else meta.dimension_slots.get(video_type, [])
                )

            infos.append(
                EvaluatorInfo(
                    meta=meta,
                    config=config_section,
                    effective_slots=effective_slots,
                )
            )
        return infos

    # ==================================================================
    # Step 4: Compute available fields
    # ==================================================================

    def _compute_available_fields(self, product_info: ProductInfo | None) -> set[str]:
        """Compute the set of fields theoretically available.

        Static layer: device-satisfied + enabled extractors' provides union.
        Runtime layer: video_path/video_type always; product_info if not None.
        """
        available: set[str] = set()

        # Runtime layer (always available)
        available.add("video_path")
        available.add("video_type")
        if product_info is not None:
            available.add("product_info")

        # Static layer: extractors that are device-satisfied and enabled
        extractors_cfg = self.config.get("extractors", {})
        for meta in extractor_registry.list_meta():
            # Check enabled
            ext_conf = extractors_cfg.get(meta.name, {})
            if isinstance(ext_conf, dict) and not ext_conf.get("enabled", True):
                continue

            # Check device
            if not self.device_manager.satisfies(meta.device_requirement):
                continue

            # Check not broken
            if extractor_registry.is_broken(meta.name):
                continue

            available.update(meta.provides)

        return available

    # ==================================================================
    # Step 5: Filter evaluators (F1/F2/F3)
    # ==================================================================

    def _filter_evaluators(
        self, evaluators: list[EvaluatorInfo], available_fields: set[str]
    ) -> list[EvaluatorInfo]:
        """Three-pass filter: F1 (enabled), F2 (device+backend), F3 (requires).

        Modifies status/reason on skipped entries. Returns all evaluators
        (including skipped ones for placeholder generation).
        """
        for info in evaluators:
            if info.status != "pending":
                continue

            # F1: enabled check
            if not info.config.get("enabled", True):
                info.status = "skipped"
                info.reason = "disabled"
                continue

            # F1b: empty effective_slots means not applicable for this video_type
            if not info.effective_slots:
                info.status = "skipped"
                info.reason = "no_slots_for_video_type"
                continue

            # F2: device requirement
            if not self.device_manager.satisfies(info.meta.device_requirement):
                info.status = "skipped"
                info.reason = "device_unavailable"
                continue

            # F2b: backend device linkage (if evaluator declares backend_config_key)
            if info.meta.backend_config_key:
                backend_ref = info.config.get(info.meta.backend_config_key)
                if backend_ref:
                    if not backend_registry.has(backend_ref):
                        # Spec §4.3 F2: unregistered backend → config error (exit 2)
                        raise ConfigError(
                            f"Evaluator '{info.meta.name}' references backend "
                            f"'{backend_ref}' which is not registered"
                        )
                    if backend_registry.is_broken(backend_ref):
                        info.status = "skipped"
                        info.reason = "import_error"
                        continue
                    # Check backend device_requirement
                    backend_meta = backend_registry.get_meta(backend_ref)
                    if not self.device_manager.satisfies(backend_meta.device_requirement):
                        info.status = "skipped"
                        info.reason = "device_unavailable"
                        continue

            # F2c: broken evaluator
            if evaluator_registry.is_broken(info.meta.name):
                info.status = "skipped"
                info.reason = "import_error"
                continue

            # F3: requires (field dependencies)
            missing = set(info.meta.requires) - available_fields
            if missing:
                info.status = "skipped"
                # Spec §4.3 F3: differentiate product_info absence
                if missing == {"product_info"} or "product_info" in missing:
                    info.reason = "missing_product_info"
                else:
                    info.reason = "missing_dependency"
                logger.debug(
                    "Evaluator '%s' skipped: missing fields %s",
                    info.meta.name,
                    missing,
                )
                continue

        return evaluators

    # ==================================================================
    # Step 6: Run extractors
    # ==================================================================

    def _run_extractors(
        self,
        video_path: str,
        product_info: ProductInfo | None,
        video_type: str,
        filtered: list[EvaluatorInfo],
    ) -> EvalContext:
        """Compute required_keys, closure expansion, topo sort, run serially."""
        # Build context
        context = EvalContext(
            video_path=video_path,
            video_type=video_type,
            product_info=product_info,
        )

        # Compute required_keys from all pending evaluators
        required_keys: set[str] = set()
        for info in filtered:
            if info.status == "pending":
                required_keys.update(info.meta.requires)

        # Subtract base fields (already in context)
        base_fields = {"video_path", "video_type"}
        if product_info is not None:
            base_fields.add("product_info")
        required_keys -= base_fields

        if not required_keys:
            return context

        # Select candidate extractors (device + enabled + not broken)
        extractors_cfg = self.config.get("extractors", {})
        candidate_metas: list[PluginMeta] = []
        for meta in extractor_registry.list_meta():
            ext_conf = extractors_cfg.get(meta.name, {})
            if isinstance(ext_conf, dict) and not ext_conf.get("enabled", True):
                continue
            if not self.device_manager.satisfies(meta.device_requirement):
                continue
            if extractor_registry.is_broken(meta.name):
                continue
            candidate_metas.append(meta)

        # Build seed: extractors whose provides intersects required_keys
        seed = [m for m in candidate_metas if set(m.provides) & required_keys]

        # Closure expansion
        try:
            closure = self._expand_closure(seed, candidate_metas, base_fields)
        except ConfigError:
            raise

        # Topo sort
        sorted_extractors = self._topo_sort(closure)

        # Serial execution with context manager
        for ext_meta in sorted_extractors:
            try:
                cls = extractor_registry.get(ext_meta.name)
            except MaterializationError as exc:
                if ext_meta.criticality == "required":
                    raise ExtractionError(ext_meta.name) from exc
                # Optional: record failure
                for field in ext_meta.provides:
                    context.extraction_failures[field] = f"materialize_failed: {exc}"
                continue

            ext_config = extractors_cfg.get(ext_meta.name, {})
            if not isinstance(ext_config, dict):
                ext_config = {}

            try:
                instance = cls(self.device_manager, ext_config)
                instance.__enter__()
            except Exception as exc:
                if ext_meta.criticality == "required":
                    raise ExtractionError(ext_meta.name) from exc
                for field in ext_meta.provides:
                    context.extraction_failures[field] = f"init_failed: {exc}"
                continue

            try:
                feats = instance.extract(context.readonly())
                context.merge(feats, ext_meta.provides)
            except Exception as exc:
                if ext_meta.criticality == "required":
                    try:
                        instance.__exit__(None, None, None)
                    except Exception:
                        pass
                    raise ExtractionError(ext_meta.name) from exc
                for field in ext_meta.provides:
                    context.extraction_failures[field] = (
                        f"extraction_failed: {exc}"
                    )
            finally:
                try:
                    instance.__exit__(None, None, None)
                except Exception:
                    pass

        return context

    def _expand_closure(
        self,
        seed: list[PluginMeta],
        extractor_metas: list[PluginMeta],
        base_fields: set[str],
    ) -> list[PluginMeta]:
        """Expand extractor closure: iteratively add providers for unmet deps."""
        selected: dict[str, PluginMeta] = {m.name: m for m in seed}

        for _iteration in range(50):  # guard against infinite loops
            satisfied = set(base_fields)
            for m in selected.values():
                satisfied.update(m.provides)

            pending: set[str] = set()
            for m in selected.values():
                pending.update(set(m.requires) - satisfied)

            if not pending:
                return list(selected.values())

            added = False
            for key in list(pending):
                provider = next(
                    (m for m in extractor_metas if key in m.provides),
                    None,
                )
                if provider is None:
                    raise ConfigError(
                        f"Extractor dependency field '{key}' has no provider"
                    )
                if provider.name not in selected:
                    selected[provider.name] = provider
                    added = True

            if not added:
                # All providers already selected but deps still unmet
                raise ConfigError(
                    f"Circular or unresolvable extractor dependencies: {pending}"
                )

        raise ConfigError("Extractor closure expansion exceeded iteration limit")

    def _topo_sort(self, extractors: list[PluginMeta]) -> list[PluginMeta]:
        """Topological sort: extractor A requires field provided by B -> B before A.

        Uses Kahn's algorithm. Raises ConfigError on cycles.
        """
        if not extractors:
            return []

        # Build provides -> extractor name mapping
        field_provider: dict[str, str] = {}
        for m in extractors:
            for field in m.provides:
                field_provider[field] = m.name

        # Build adjacency and in-degree
        name_to_meta: dict[str, PluginMeta] = {m.name: m for m in extractors}
        in_degree: dict[str, int] = {m.name: 0 for m in extractors}
        adj: dict[str, list[str]] = {m.name: [] for m in extractors}

        for m in extractors:
            for req in m.requires:
                provider_name = field_provider.get(req)
                if provider_name and provider_name in name_to_meta:
                    # edge: provider -> m (provider must run first)
                    adj[provider_name].append(m.name)
                    in_degree[m.name] += 1

        # Kahn's algorithm
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: list[PluginMeta] = []

        while queue:
            node = queue.pop(0)
            result.append(name_to_meta[node])
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(extractors):
            raise ConfigError(
                "Cycle detected in extractor dependency graph: "
                f"resolved {len(result)}/{len(extractors)}"
            )

        return result

    # ==================================================================
    # Step 7: Re-filter after extraction
    # ==================================================================

    def _refilter_after_extraction(
        self, filtered: list[EvaluatorInfo], context: EvalContext
    ) -> list[EvaluatorInfo]:
        """Check extraction_failures; mark affected evaluators as skipped."""
        if not context.extraction_failures:
            return filtered

        failed_fields = set(context.extraction_failures.keys())

        for info in filtered:
            if info.status != "pending":
                continue
            required_fields = set(info.meta.requires)
            if required_fields & failed_fields:
                info.status = "skipped"
                info.reason = "extraction_failed"
                logger.debug(
                    "Evaluator '%s' skipped after extraction: depends on failed "
                    "fields %s",
                    info.meta.name,
                    required_fields & failed_fields,
                )

        return filtered

    # ==================================================================
    # Step 8: Run evaluators
    # ==================================================================

    def _run_evaluators(
        self, context: EvalContext, filtered: list[EvaluatorInfo]
    ) -> dict[str, EvalResult]:
        """Execute evaluators serially with context managers."""
        results: dict[str, EvalResult] = {}

        for info in filtered:
            if info.status != "pending":
                continue  # already skipped

            # Get class from registry
            try:
                cls = evaluator_registry.get(info.meta.name)
            except MaterializationError:
                info.status = "skipped"
                info.reason = "import_error"
                continue

            # Build config for evaluator instance
            eval_config = dict(info.config)
            eval_config["_backends"] = self.config.get("backends", {})

            # Instantiate
            try:
                instance = cls(self.device_manager, eval_config)
            except Exception as exc:
                info.status = "error"
                info.reason = "init_failed"
                logger.warning(
                    "Evaluator '%s' instantiation failed: %s", info.meta.name, exc
                )
                continue

            # __enter__
            try:
                instance.__enter__()
            except Exception as exc:
                info.status = "error"
                info.reason = "init_failed"
                logger.warning(
                    "Evaluator '%s' __enter__ failed: %s", info.meta.name, exc
                )
                continue

            # evaluate
            try:
                available, reason = instance.check_availability()
                if not available:
                    info.status = "skipped"
                    info.reason = "runtime_unavailable"
                    logger.debug(
                        "Evaluator '%s' runtime unavailable: %s",
                        info.meta.name,
                        reason,
                    )
                    continue

                result = instance.evaluate(context.readonly())

                # result can be EvalResult or list[EvalResult]
                if isinstance(result, list):
                    for r in result:
                        results[r.dimension] = r
                else:
                    results[result.dimension] = result

                info.status = "active"
            except Exception as exc:
                info.status = "error"
                info.reason = "evaluation_failed"
                logger.warning(
                    "Evaluator '%s' evaluation failed: %s", info.meta.name, exc
                )
            finally:
                try:
                    instance.__exit__(None, None, None)
                except Exception:
                    pass

        return results

    # ==================================================================
    # Step 9: Fill placeholders
    # ==================================================================

    def _fill_placeholders(
        self,
        results: dict[str, EvalResult],
        evaluators: list[EvaluatorInfo],
        filtered: list[EvaluatorInfo],
        video_type: str,
    ) -> dict[str, EvalResult]:
        """Generate placeholder EvalResults for skipped/error evaluators.

        Expands multi-slot evaluators into all their effective_slots.
        Disabled evaluators (F1) do NOT generate placeholders.
        """
        for info in evaluators:
            # Skip active evaluators (already have results) and disabled ones
            if info.status == "active":
                continue
            if info.status == "skipped" and info.reason == "disabled":
                continue
            if info.status == "skipped" and info.reason == "no_slots_for_video_type":
                continue
            if info.status == "pending":
                # Still pending means it was never attempted (shouldn't happen
                # normally, but be defensive)
                continue

            # Generate placeholder for each effective_slot
            status = info.status if info.status in ("skipped", "error") else "skipped"
            for slot in info.effective_slots:
                if slot not in results:
                    results[slot] = EvalResult(
                        dimension=slot,
                        evaluator=info.meta.name,
                        score=0.0,
                        status=status,
                        reason=info.reason,
                    )

        return results

    # ==================================================================
    # Step 10: Build default weights
    # ==================================================================

    def _build_default_weights(
        self, evaluators: list[EvaluatorInfo]
    ) -> dict[str, float | None]:
        """Build default_weights dict from evaluator metadata.

        For single-slot evaluators: {name: default_weights} (float or None).
        For multi-slot evaluators with dict default_weights: expand per-slot.
        """
        weights: dict[str, float | None] = {}

        for info in evaluators:
            dw = info.meta.default_weights

            if dw is None:
                # No declared default weight: all slots get None
                for slot in info.effective_slots:
                    weights.setdefault(slot, None)
            elif isinstance(dw, (int, float)):
                # Single weight applies to all slots (split evenly if multi-slot)
                if len(info.effective_slots) == 1:
                    weights.setdefault(info.effective_slots[0], float(dw))
                else:
                    per_slot = float(dw) / len(info.effective_slots)
                    for slot in info.effective_slots:
                        weights.setdefault(slot, per_slot)
            elif isinstance(dw, dict):
                # Per-slot weights
                for slot in info.effective_slots:
                    weights.setdefault(slot, dw.get(slot))

        return weights

    # ==================================================================
    # Step 11: Fuse
    # ==================================================================

    def _fuse(
        self,
        results: dict[str, EvalResult],
        video_type: str,
        default_weights: dict[str, float | None],
    ) -> FusionOutcome:
        """Get fusion strategy and fuse results."""
        strategy_name = self.config.get("fusion", {}).get("strategy", "weighted_veto")
        fusion_cls = fusion_registry.get(strategy_name)
        fusion = fusion_cls(self.config.get("fusion", {}))
        weights = self.config.get("fusion", {}).get(f"weights_{video_type}", {})
        return fusion.fuse(results, video_type, weights, default_weights)

    # ==================================================================
    # Step 12: Assemble report
    # ==================================================================

    def _assemble_report(
        self,
        outcome: FusionOutcome,
        video_path: str,
        video_type: str,
        results: dict[str, EvalResult],
        evaluators: list[EvaluatorInfo],
    ) -> FinalReport:
        """Build FinalReport with ReportMeta."""
        # Collect evaluator versions
        evaluator_versions: dict[str, str] = {}
        for info in evaluators:
            if info.status == "active":
                evaluator_versions[info.meta.name] = info.meta.version

        # Collect skipped evaluator names
        skipped = [
            info.meta.name
            for info in evaluators
            if info.status in ("skipped", "error")
            and info.reason != "disabled"
            and info.reason != "no_slots_for_video_type"
        ]

        # Determine backend info
        backend_name = "none"
        vlm_model = "none"
        evaluators_cfg = self.config.get("evaluators", {})
        for eval_name, eval_conf in evaluators_cfg.items():
            if isinstance(eval_conf, dict) and eval_conf.get("backend"):
                backend_name = eval_conf["backend"]
                backends_cfg = self.config.get("backends", {})
                backend_conf = backends_cfg.get(backend_name, {})
                vlm_model = backend_conf.get("model", "unknown")
                break

        meta = ReportMeta(
            framework_version=__version__,
            device=self.device_manager.device_type,
            backend=backend_name,
            vlm_model=vlm_model,
            evaluator_versions=evaluator_versions,
            skipped=skipped,
            config_hash=compute_config_hash(self.config),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return FinalReport(
            video_path=video_path,
            video_type=video_type,
            overall_score=outcome.overall_score,
            grade=outcome.grade,
            passed=outcome.passed,
            veto_reasons=outcome.veto_reasons,
            dimension_results=results,
            suggestions=outcome.suggestions,
            meta=meta,
        )
