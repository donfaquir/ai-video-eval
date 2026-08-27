"""Configuration loading, merging, validation, and hashing."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from video_eval.core.exceptions import ConfigError
from video_eval.core.schemas import ValidationIssue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration (used when config.yaml is absent)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "device": {"preferred": "auto"},
    "extractors": {
        "fps": 1,
        "max_frames": 64,
        "asr": {"enabled": True, "model_size": "large-v3", "language": "auto", "beam_size": 5},
        "ocr": {"confidence": 0.5},
        "clip_features": {"model_name": "ViT-SO400M-14-SigLIP-384"},
    },
    "backends": {
        "local": {"model": "Qwen/Qwen3-VL-8B-Instruct"},
        "api": {
            "provider": "gemini",
            "model": "gemini-3-flash",
            "timeout": 30,
            "max_retries": 3,
            "retry_base": 1.0,
        },
        "mock": {},
    },
    "evaluators": {
        "technical_quality": {"enabled": True},
        "aigc_defect": {"enabled": True, "model": "openai/clip-vit-large-patch14", "defect_threshold": 0.6},
        "product_fidelity": {"enabled": True},
        "compliance": {
            "enabled": True,
            "limit_words": ["最", "第一", "国家级", "顶级", "极品", "全网最低", "独家"],
            "medical_words": ["治疗", "疗效", "药到病除", "根治"],
            "banned_entities": [],
        },
        "vlm_judge": {
            "enabled": True,
            "backend": "mock",
            "api_concurrency": 4,
            "api_max_failures": 5,
            "dimensions_main_image": ["sellpoint_coverage", "cross_modal"],
            "dimensions_external": ["hook_strength", "marketing_logic", "audience_match", "cross_modal"],
            "dimensions_general": ["cross_modal"],
        },
    },
    "fusion": {
        "strategy": "weighted_veto",
        "strict_veto_dims": ["compliance"],
        "thresholds": {"A": 0.75, "B": 0.60, "C": 0.40},
        "veto_thresholds": {"compliance": 0.0, "product_fidelity": 0.3, "aigc_defect": 0.3},
        "weights_main_image": {
            "technical_quality": 0.15,
            "aigc_defect": 0.15,
            "product_fidelity": 0.20,
            "sellpoint_coverage": 0.20,
            "cross_modal": 0.10,
        },
        "weights_external": {
            "technical_quality": 0.10,
            "aigc_defect": 0.10,
            "product_fidelity": 0.10,
            "hook_strength": 0.20,
            "marketing_logic": 0.15,
            "audience_match": 0.10,
            "cross_modal": 0.05,
        },
        "weights_general": {
            "technical_quality": 0.35,
            "aigc_defect": 0.35,
            "cross_modal": 0.30,
        },
    },
    "batch": {"mode": "resident", "chunk_size": 8},
    "output": {"format": "json", "include_meta": True, "include_evidence": True, "pretty": True},
}


# ---------------------------------------------------------------------------
# Utility: deep merge dicts
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Three-layer config merging: CLI > config.yaml > plugin defaults."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the loader.

        Args:
            config_path: Path to config.yaml. None defaults to ./config.yaml.
        """
        self.config_path = config_path or "config.yaml"

    def load(self) -> dict:
        """Read config.yaml and merge with plugin defaults.

        - File not found: use all-defaults + log warning.
        - YAML parse error: raise ConfigError.

        Returns:
            Merged configuration dictionary.
        """
        path = Path(self.config_path)
        if not path.exists():
            logger.warning(
                "Config file '%s' not found; using all defaults.", self.config_path
            )
            file_config: dict = {}
        else:
            try:
                raw = path.read_text(encoding="utf-8")
                file_config = yaml.safe_load(raw) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"Failed to parse YAML config '{self.config_path}': {exc}") from exc

        # Layer 1: start from built-in defaults
        config = _deep_merge(_DEFAULT_CONFIG, {})

        # Layer 2: merge plugin defaults from registries
        config = self._merge_plugin_defaults(config)

        # Layer 3: merge user file config (overrides defaults)
        config = _deep_merge(config, file_config)

        return config

    def merge_cli_overrides(self, config: dict, cli_args: list[str]) -> dict:
        """Parse KEY=VALUE strings and deep-merge into config.

        Type conversion:
          1. Schema-driven: if key path leads to a plugin config_schema, force that type.
          2. Heuristic fallback: int -> float -> bool -> comma-split list -> str.

        Args:
            config: Base configuration to merge into.
            cli_args: List of "dotted.key.path=value" strings.

        Returns:
            Updated configuration dictionary.
        """
        result = _deep_merge(config, {})

        for arg in cli_args:
            if "=" not in arg:
                raise ConfigError(f"Invalid --set argument (missing '='): '{arg}'")

            key_path, raw_value = arg.split("=", 1)
            parts = key_path.split(".")
            converted = self._convert_value(key_path, raw_value, result)

            # Deep-set the value at dotted path
            target = result
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = converted

        return result

    def validate(self, config: dict) -> list[ValidationIssue]:
        """Full validation of the configuration.

        Returns a list of ValidationIssue (error/warning).
        Errors should cause exit code 2; warnings are informational.
        """
        issues: list[ValidationIssue] = []

        # Import registries at function level to avoid circular imports
        from video_eval.core.registry import (
            backend_registry,
            evaluator_registry,
            extractor_registry,
        )

        # Rule 1: Evaluator enabled=true but not registered
        evaluators_cfg = config.get("evaluators", {})
        for eval_name, eval_conf in evaluators_cfg.items():
            if not isinstance(eval_conf, dict):
                continue
            if eval_conf.get("enabled", True):
                if not evaluator_registry.has(eval_name):
                    issues.append(ValidationIssue(
                        severity="error",
                        plugin_name=eval_name,
                        message=f"Evaluator '{eval_name}' is enabled but not found in registry.",
                        field="evaluators",
                    ))

        # Rule 2: Required fields missing
        # Check required fields per evaluator config_schema
        for eval_name, eval_conf in evaluators_cfg.items():
            if not isinstance(eval_conf, dict):
                continue
            if not eval_conf.get("enabled", True):
                continue
            if not evaluator_registry.has(eval_name):
                continue
            meta = evaluator_registry.get_meta(eval_name)
            schema = meta.config_schema
            for field_name, field_spec in schema.items():
                if isinstance(field_spec, dict) and field_spec.get("required", False):
                    if field_name not in eval_conf:
                        issues.append(ValidationIssue(
                            severity="error",
                            plugin_name=eval_name,
                            message=f"Required field '{field_name}' missing in evaluator '{eval_name}' config.",
                            field=field_name,
                        ))

        # Rule 3: Unknown keys in evaluator config section
        for eval_name, eval_conf in evaluators_cfg.items():
            if not isinstance(eval_conf, dict):
                continue
            if not evaluator_registry.has(eval_name):
                continue
            meta = evaluator_registry.get_meta(eval_name)
            schema = meta.config_schema
            # Standard keys that are always allowed
            standard_keys = {"enabled"}
            known_keys = standard_keys | set(schema.keys())
            # Also allow dimension override keys (dimensions_*)
            for key in list(eval_conf.keys()):
                if key.startswith("dimensions_"):
                    known_keys.add(key)
            for key in eval_conf:
                if key not in known_keys:
                    issues.append(ValidationIssue(
                        severity="warning",
                        plugin_name=eval_name,
                        message=f"Unknown key '{key}' in evaluator '{eval_name}' config.",
                        field=key,
                    ))

        # Rule 4: Unknown keys in backend config section
        backends_cfg = config.get("backends", {})
        for backend_name, backend_conf in backends_cfg.items():
            if not isinstance(backend_conf, dict):
                continue
            if not backend_registry.has(backend_name):
                # Rule 5 handles unregistered backends
                continue
            meta = backend_registry.get_meta(backend_name)
            schema = meta.config_schema
            known_keys = set(schema.keys())
            for key in backend_conf:
                if key not in known_keys:
                    issues.append(ValidationIssue(
                        severity="warning",
                        plugin_name=backend_name,
                        message=f"Unknown key '{key}' in backend '{backend_name}' config.",
                        field=key,
                    ))

        # Rule 5: Backend referenced but not registered (when selected by evaluator)
        for eval_name, eval_conf in evaluators_cfg.items():
            if not isinstance(eval_conf, dict):
                continue
            if not eval_conf.get("enabled", True):
                continue
            backend_ref = eval_conf.get("backend")
            if backend_ref and not backend_registry.has(backend_ref):
                issues.append(ValidationIssue(
                    severity="error",
                    plugin_name=eval_name,
                    message=(
                        f"Evaluator '{eval_name}' references backend '{backend_ref}' "
                        f"which is not registered."
                    ),
                    field="backend",
                ))

        # Rule 6: strict_veto_dims dimension disabled/unregistered
        fusion_cfg = config.get("fusion", {})
        strict_veto_dims = fusion_cfg.get("strict_veto_dims", [])
        for dim in strict_veto_dims:
            # Check if any enabled evaluator can produce this dimension
            dim_available = False
            for eval_name, eval_conf in evaluators_cfg.items():
                if not isinstance(eval_conf, dict):
                    continue
                if not eval_conf.get("enabled", True):
                    continue
                if not evaluator_registry.has(eval_name):
                    continue
                meta = evaluator_registry.get_meta(eval_name)
                # Single-slot evaluator: dimension = evaluator name
                if meta.dimension_slots is None:
                    if eval_name == dim:
                        dim_available = True
                        break
                else:
                    # Multi-slot: check all possible slot lists
                    for slots in meta.dimension_slots.values():
                        if dim in slots:
                            dim_available = True
                            break
                    if dim_available:
                        break
            if not dim_available:
                issues.append(ValidationIssue(
                    severity="error",
                    plugin_name="fusion",
                    message=(
                        f"strict_veto_dims dimension '{dim}' is not produced by "
                        f"any enabled evaluator."
                    ),
                    field="strict_veto_dims",
                ))

        # Rule 7: strict_veto_dims dependency closure unreachable
        # TODO: Implement D9 probe integration after Pipeline is available.

        # Rule 8: backend_config_key points to unregistered backend
        for eval_name, eval_conf in evaluators_cfg.items():
            if not isinstance(eval_conf, dict):
                continue
            if not eval_conf.get("enabled", True):
                continue
            if not evaluator_registry.has(eval_name):
                continue
            meta = evaluator_registry.get_meta(eval_name)
            if meta.backend_config_key:
                if not backend_registry.has(meta.backend_config_key):
                    issues.append(ValidationIssue(
                        severity="error",
                        plugin_name=eval_name,
                        message=(
                            f"Evaluator '{eval_name}' declares backend_config_key "
                            f"'{meta.backend_config_key}' which is not registered."
                        ),
                        field="backend_config_key",
                    ))

        # Rule 9: resident mode memory estimation
        # TODO: Implement memory pre-check after DeviceManager VRAM queries are available.

        # Rule 10: Required extractor disabled
        for ext_name in extractor_registry.list():
            ext_meta = extractor_registry.get_meta(ext_name)
            if ext_meta.criticality == "required":
                ext_cfg = config.get("extractors", {}).get(ext_name, {})
                if isinstance(ext_cfg, dict) and not ext_cfg.get("enabled", True):
                    issues.append(ValidationIssue(
                        severity="warning",
                        plugin_name=ext_name,
                        message=(
                            f"Required extractor '{ext_name}' is disabled; "
                            f"fields it provides will be unavailable."
                        ),
                        field="enabled",
                    ))

        # Rule 11: Enabled evaluator with no weight and not in veto
        veto_thresholds = fusion_cfg.get("veto_thresholds", {})
        all_weight_maps = {}
        for key in fusion_cfg:
            if key.startswith("weights_"):
                wmap = fusion_cfg[key]
                if isinstance(wmap, dict):
                    all_weight_maps.update(wmap)
        for eval_name, eval_conf in evaluators_cfg.items():
            if not isinstance(eval_conf, dict):
                continue
            if not eval_conf.get("enabled", True):
                continue
            if eval_name not in all_weight_maps and eval_name not in veto_thresholds and eval_name not in strict_veto_dims:
                # Check if it's a multi-slot evaluator whose slots appear in weights
                if evaluator_registry.has(eval_name):
                    meta = evaluator_registry.get_meta(eval_name)
                    if meta.dimension_slots is not None:
                        # Multi-slot: check if any slot has weight
                        has_weight = False
                        for slots in meta.dimension_slots.values():
                            for slot in slots:
                                if slot in all_weight_maps or slot in veto_thresholds or slot in strict_veto_dims:
                                    has_weight = True
                                    break
                            if has_weight:
                                break
                        if has_weight:
                            continue
                issues.append(ValidationIssue(
                    severity="warning",
                    plugin_name=eval_name,
                    message=(
                        f"Evaluator '{eval_name}' is enabled but has no weight "
                        f"and is not a veto dimension."
                    ),
                    field="weights",
                ))

        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_plugin_defaults(self, config: dict) -> dict:
        """Merge default values from plugin config_schemas into config.

        For each registered evaluator/extractor/backend, read its config_schema.
        For each key with a 'default' value, set it in config if not already present.
        """
        # Import at function level to avoid circular imports
        from video_eval.core.registry import (
            backend_registry,
            evaluator_registry,
            extractor_registry,
        )

        result = _deep_merge(config, {})

        # Evaluator defaults
        evaluators_cfg = result.setdefault("evaluators", {})
        for eval_name in evaluator_registry.list():
            meta = evaluator_registry.get_meta(eval_name)
            if not meta.config_schema:
                continue
            eval_section = evaluators_cfg.setdefault(eval_name, {})
            if not isinstance(eval_section, dict):
                continue
            for field_name, field_spec in meta.config_schema.items():
                if isinstance(field_spec, dict) and "default" in field_spec:
                    eval_section.setdefault(field_name, field_spec["default"])

        # Extractor defaults
        extractors_cfg = result.setdefault("extractors", {})
        for ext_name in extractor_registry.list():
            meta = extractor_registry.get_meta(ext_name)
            if not meta.config_schema:
                continue
            ext_section = extractors_cfg.setdefault(ext_name, {})
            if not isinstance(ext_section, dict):
                continue
            for field_name, field_spec in meta.config_schema.items():
                if isinstance(field_spec, dict) and "default" in field_spec:
                    ext_section.setdefault(field_name, field_spec["default"])

        # Backend defaults
        backends_cfg = result.setdefault("backends", {})
        for backend_name in backend_registry.list():
            meta = backend_registry.get_meta(backend_name)
            if not meta.config_schema:
                continue
            backend_section = backends_cfg.setdefault(backend_name, {})
            if not isinstance(backend_section, dict):
                continue
            for field_name, field_spec in meta.config_schema.items():
                if isinstance(field_spec, dict) and "default" in field_spec:
                    backend_section.setdefault(field_name, field_spec["default"])

        return result

    def _convert_value(self, key_path: str, raw_value: str, config: dict) -> Any:
        """Convert a raw string value to the appropriate type.

        1. Check plugin config_schema for declared type (schema-driven).
        2. Heuristic fallback: int -> float -> bool -> comma-split list -> str.

        Raises:
            ConfigError: If schema-driven conversion fails.
        """
        # Try schema-driven conversion
        schema_type = self._lookup_schema_type(key_path)
        if schema_type is not None:
            return self._force_schema_type(raw_value, schema_type, key_path)

        # Heuristic fallback
        return self._heuristic_convert(raw_value)

    def _lookup_schema_type(self, key_path: str) -> str | None:
        """Look up the declared type for a key path in plugin config schemas.

        Returns the type string ('int', 'float', 'str', 'bool', 'list') or None.
        """
        from video_eval.core.registry import (
            backend_registry,
            evaluator_registry,
            extractor_registry,
        )

        parts = key_path.split(".")
        if len(parts) < 2:
            return None

        section = parts[0]
        plugin_name = parts[1]
        field_name = parts[-1] if len(parts) > 2 else None

        if field_name is None:
            return None

        registry = None
        if section == "evaluators":
            registry = evaluator_registry
        elif section == "extractors":
            registry = extractor_registry
        elif section == "backends":
            registry = backend_registry

        if registry is None or not registry.has(plugin_name):
            return None

        meta = registry.get_meta(plugin_name)
        field_spec = meta.config_schema.get(field_name)
        if isinstance(field_spec, dict) and "type" in field_spec:
            return field_spec["type"]

        return None

    def _force_schema_type(self, raw_value: str, schema_type: str, key_path: str) -> Any:
        """Force conversion to a schema-declared type.

        Raises:
            ConfigError: If conversion fails.
        """
        try:
            if schema_type == "int":
                return int(raw_value)
            elif schema_type == "float":
                return float(raw_value)
            elif schema_type == "bool":
                if raw_value.lower() in ("true", "1", "yes"):
                    return True
                elif raw_value.lower() in ("false", "0", "no"):
                    return False
                raise ValueError(f"Cannot convert '{raw_value}' to bool")
            elif schema_type == "list":
                return [item.strip() for item in raw_value.split(",")]
            elif schema_type == "str":
                return raw_value
            else:
                return raw_value
        except (ValueError, TypeError) as exc:
            raise ConfigError(
                f"Cannot convert --set value for '{key_path}' to {schema_type}: "
                f"'{raw_value}' ({exc})"
            ) from exc

    def _heuristic_convert(self, raw_value: str) -> Any:
        """Heuristic type conversion: int -> float -> bool -> list -> str."""
        # Try int
        try:
            return int(raw_value)
        except ValueError:
            pass

        # Try float
        try:
            return float(raw_value)
        except ValueError:
            pass

        # Try bool
        if raw_value.lower() == "true":
            return True
        if raw_value.lower() == "false":
            return False

        # Try comma-split list (only if comma present)
        if "," in raw_value:
            return [item.strip() for item in raw_value.split(",")]

        # Default: string
        return raw_value


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------


def compute_config_hash(config: dict) -> str:
    """Canonical serialization (sorted keys) -> SHA-256 -> first 8 hex chars.

    Key order does not affect the result.
    Excludes runtime info (device detection, timestamps) if present.
    """
    # Strip runtime-only keys that shouldn't affect the hash
    hashable = {k: v for k, v in config.items() if k not in ("_runtime",)}
    canonical = json.dumps(hashable, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
