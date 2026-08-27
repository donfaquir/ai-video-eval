"""Plugin registry system: Registry[T], Placeholder, decorators, and discovery."""

from __future__ import annotations

import importlib
import logging
import pkgutil
import threading
from typing import Any, Callable, Generic, TypeVar

from video_eval.core.exceptions import (
    DuplicateRegistrationError,
    MaterializationError,
    NameNotFoundError,
    RegistryError,
    RegistryFrozenError,
)
from video_eval.core.schemas import PluginMeta

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Levenshtein distance (no external dependency)
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(
                min(
                    curr_row[j] + 1,         # insert
                    prev_row[j + 1] + 1,     # delete
                    prev_row[j] + cost,      # substitute
                )
            )
        prev_row = curr_row
    return prev_row[-1]


# ---------------------------------------------------------------------------
# Placeholder
# ---------------------------------------------------------------------------


class Placeholder(Generic[T]):
    """A lazy-loading stand-in for a plugin class that has not yet been imported."""

    def __init__(
        self,
        name: str,
        loader: Callable[[], type[T]],
        *,
        origin: str,
        module_path: str | None = None,
    ) -> None:
        self.name = name
        self.loader = loader
        self.origin = origin
        self.module_path = module_path


# ---------------------------------------------------------------------------
# Registry[T]
# ---------------------------------------------------------------------------


class Registry(Generic[T]):
    """Generic, thread-safe registry mapping alias -> plugin class or Placeholder."""

    def __init__(
        self,
        name: str,
        base_cls: type[T] | None,
        *,
        allow_override: bool = False,
    ) -> None:
        self.name = name
        self.base_cls = base_cls
        self.allow_override = allow_override
        self._entries: dict[str, type[T] | Placeholder[T]] = {}
        self._meta: dict[str, PluginMeta] = {}
        self._origins: dict[str, str] = {}
        self._broken: dict[str, Exception] = {}
        self._frozen: bool = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        cls: type[T] | None = None,
        origin: str = "explicit",
    ) -> Callable | None:
        """Register a class under *name*.

        If *cls* is None, return a decorator that will perform the registration.
        """
        if cls is None:
            # Return decorator form
            def decorator(cls_inner: type[T]) -> type[T]:
                self.register(name, cls=cls_inner, origin=origin)
                return cls_inner
            return decorator

        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(
                    f"Registry '{self.name}' is frozen; cannot register '{name}'",
                    self.name,
                )

            is_new_placeholder = isinstance(cls, Placeholder)
            existing = self._entries.get(name)

            if existing is not None:
                existing_is_placeholder = isinstance(existing, Placeholder)

                if existing_is_placeholder and not is_new_placeholder:
                    # Placeholder -> Concrete: allow upgrade
                    pass
                elif not existing_is_placeholder and not is_new_placeholder:
                    # Concrete -> Concrete
                    if self.allow_override:
                        pass  # allow override
                    else:
                        raise DuplicateRegistrationError(
                            f"'{name}' already registered in '{self.name}' registry "
                            f"(origin='{self._origins[name]}'). "
                            f"Set allow_override=True to permit.",
                            self.name,
                        )
                elif not existing_is_placeholder and is_new_placeholder:
                    # Concrete -> Placeholder: reject
                    raise DuplicateRegistrationError(
                        f"'{name}' already registered as concrete in '{self.name}' "
                        f"registry; cannot downgrade to Placeholder.",
                        self.name,
                    )
                else:
                    # Placeholder -> Placeholder: keep first
                    return None

            # Perform registration
            if is_new_placeholder:
                placeholder: Placeholder[T] = cls  # type: ignore[assignment]
                self._entries[name] = placeholder
                self._meta[name] = PluginMeta(
                    name=name,
                    version="0.0",
                    device_requirement="any",
                    requires=[],
                    config_schema={},
                    provides=[],
                    criticality="optional",
                    backend_config_key=None,
                    default_weights=None,
                    dimension_slots=None,
                    origin=origin,
                    is_placeholder=True,
                )
                self._origins[name] = origin
            else:
                # Concrete class
                if self.base_cls is not None and not issubclass(cls, self.base_cls):
                    raise RegistryError(
                        f"'{name}' class {cls!r} is not a subclass of "
                        f"{self.base_cls!r}",
                        self.name,
                    )
                self._entries[name] = cls
                self._meta[name] = _extract_meta(cls, name, origin)
                self._origins[name] = origin

        return None

    def get(self, name: str) -> type[T]:
        """Retrieve the plugin class by name, materializing if necessary."""
        # Fast path: check broken (no lock needed for set membership read)
        if name in self._broken:
            original = self._broken[name]
            raise MaterializationError(
                f"'{name}' previously failed to materialize in '{self.name}' "
                f"registry: {original}",
                self.name,
            )

        entry = self._entries.get(name)
        if entry is None:
            suggestions = self._suggest_similar(name)
            available = list(self._entries.keys())
            avail_display = available[:10]
            avail_str = ", ".join(avail_display)
            if len(available) > 10:
                avail_str += f" ... ({len(avail_display)} of {len(available)}, use list() for full list)"

            msg = f"'{name}' not found in {self.name} registry."
            if suggestions:
                msg += f"\n  Did you mean: {', '.join(repr(s) for s in suggestions)}?"
            if available:
                msg += f"\n  Available: {avail_str}"
            raise NameNotFoundError(msg, self.name)

        if isinstance(entry, Placeholder):
            return self._materialize(name, entry)

        return entry  # type: ignore[return-value]

    def get_meta(self, name: str) -> PluginMeta:
        """Get metadata for a plugin without materializing it."""
        if name not in self._meta:
            raise NameNotFoundError(
                f"'{name}' not found in {self.name} registry.",
                self.name,
            )
        return self._meta[name]

    def has(self, name: str) -> bool:
        """Check if *name* is registered (including placeholders)."""
        return name in self._entries

    def list(self) -> list[str]:
        """Return all registered names in registration order."""
        return list(self._entries.keys())

    def list_meta(self) -> list[PluginMeta]:
        """Return all PluginMeta instances in registration order."""
        return list(self._meta.values())

    def freeze(self) -> None:
        """Freeze the registry; subsequent register() calls will raise."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the registry (intended for tests only)."""
        self._frozen = False

    def origin(self, name: str) -> str:
        """Return the origin tag for a registered name."""
        if name not in self._origins:
            raise NameNotFoundError(
                f"'{name}' not found in {self.name} registry.",
                self.name,
            )
        return self._origins[name]

    def is_placeholder(self, name: str) -> bool:
        """Check whether an entry is still an unmaterialized Placeholder."""
        return isinstance(self._entries.get(name), Placeholder)

    def is_broken(self, name: str) -> bool:
        """Check whether materialization has previously failed for *name*."""
        return name in self._broken

    # ------------------------------------------------------------------
    # Materialization (8-step logic)
    # ------------------------------------------------------------------

    def _materialize(self, name: str, placeholder: Placeholder[T]) -> type[T]:
        """Materialize a Placeholder into a real class.

        Steps:
        1. Check _broken -> raise immediately (no retry)
        2. Acquire lock
        3. Double-check (another thread may have completed materialization)
        4. Call loader()
        5. On failure -> add to _broken, raise MaterializationError
        6. issubclass check (if base_cls set) -> fail -> MaterializationError
        7. Replace entry, update meta (is_placeholder=False, backfill attrs)
        8. Release lock
        """
        # Step 1: already checked in get(), but re-check for safety
        if name in self._broken:
            original = self._broken[name]
            raise MaterializationError(
                f"'{name}' previously failed to materialize in '{self.name}' "
                f"registry: {original}",
                self.name,
            )

        # Step 2: acquire lock
        with self._lock:
            # Step 3: double-check after acquiring lock
            entry = self._entries.get(name)
            if entry is None or not isinstance(entry, Placeholder):
                # Another thread already materialized or entry was removed
                if entry is None:
                    raise MaterializationError(
                        f"'{name}' disappeared during materialization.",
                        self.name,
                    )
                return entry  # type: ignore[return-value]

            # Also re-check broken inside lock
            if name in self._broken:
                original = self._broken[name]
                raise MaterializationError(
                    f"'{name}' previously failed to materialize in '{self.name}' "
                    f"registry: {original}",
                    self.name,
                )

            # Step 4: call loader
            try:
                loaded_cls = placeholder.loader()
            except Exception as exc:
                # Step 5: failure -> mark broken
                self._broken[name] = exc
                raise MaterializationError(
                    f"Failed to materialize '{name}' in '{self.name}' registry: {exc}",
                    self.name,
                ) from exc

            # Step 6: issubclass check
            if self.base_cls is not None:
                if not issubclass(loaded_cls, self.base_cls):
                    err = TypeError(
                        f"Loaded class {loaded_cls!r} for '{name}' is not a "
                        f"subclass of {self.base_cls!r}"
                    )
                    self._broken[name] = err
                    raise MaterializationError(
                        f"Type check failed for '{name}' in '{self.name}' "
                        f"registry: {err}",
                        self.name,
                    ) from err

            # Step 7: replace entry and update meta
            self._entries[name] = loaded_cls
            origin = self._origins.get(name, placeholder.origin)
            self._meta[name] = _extract_meta(loaded_cls, name, origin)

        # Step 8: lock released (context manager)
        return loaded_cls

    # ------------------------------------------------------------------
    # Smart error suggestions
    # ------------------------------------------------------------------

    def _suggest_similar(
        self, name: str, max_results: int = 3, threshold: int = 3
    ) -> list[str]:
        """Suggest similar registered names using prefix, substring, and Levenshtein.

        Scoring: prefix match = 3pts, substring match = 2pts,
        Levenshtein within threshold = (threshold - dist + 1) pts.
        Return top N by score (descending), only those with score > 0.
        """
        candidates: dict[str, int] = {}
        name_lower = name.lower()

        for registered in self._entries:
            reg_lower = registered.lower()
            score = 0

            # Prefix match
            if reg_lower.startswith(name_lower) or name_lower.startswith(reg_lower):
                score += 3

            # Substring match
            if name_lower in reg_lower or reg_lower in name_lower:
                score += 2

            # Levenshtein distance
            dist = _levenshtein(name_lower, reg_lower)
            if dist <= threshold:
                score += threshold - dist + 1

            if score > 0:
                candidates[registered] = score

        # Sort by score descending, then alphabetically for ties
        sorted_candidates = sorted(
            candidates.items(), key=lambda x: (-x[1], x[0])
        )
        return [c[0] for c in sorted_candidates[:max_results]]


# ---------------------------------------------------------------------------
# PluginMeta extraction helper
# ---------------------------------------------------------------------------


def _extract_meta(cls: type, name: str, origin: str) -> PluginMeta:
    """Extract PluginMeta from class attributes."""
    return PluginMeta(
        name=name,
        version=getattr(cls, "version", "0.1.0"),
        device_requirement=getattr(cls, "device_requirement", "any"),
        requires=getattr(cls, "requires", []),
        config_schema=getattr(cls, "config_schema", {}),
        provides=getattr(cls, "provides", []),
        criticality=getattr(cls, "criticality", "optional"),
        backend_config_key=getattr(cls, "backend_config_key", None),
        default_weights=getattr(cls, "default_weights", None),
        dimension_slots=getattr(cls, "dimension_slots", None),
        origin=origin,
        is_placeholder=False,
    )


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------


def register_evaluator(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator: register a class as an evaluator plugin."""

    def decorator(cls: type[T]) -> type[T]:
        if getattr(cls, "name", None) != name:
            raise RegistryError(
                f"Alias '{name}' != cls.name '{getattr(cls, 'name', None)}'",
                "evaluator",
            )
        evaluator_registry.register(name, cls=cls, origin="builtin")
        return cls

    return decorator


def register_backend(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator: register a class as a backend plugin."""

    def decorator(cls: type[T]) -> type[T]:
        if getattr(cls, "name", None) != name:
            raise RegistryError(
                f"Alias '{name}' != cls.name '{getattr(cls, 'name', None)}'",
                "backend",
            )
        backend_registry.register(name, cls=cls, origin="builtin")
        return cls

    return decorator


def register_extractor(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator: register a class as an extractor plugin.

    Also checks for provides field conflicts with existing extractors.
    """

    def decorator(cls: type[T]) -> type[T]:
        if getattr(cls, "name", None) != name:
            raise RegistryError(
                f"Alias '{name}' != cls.name '{getattr(cls, 'name', None)}'",
                "extractor",
            )
        # Check provides conflict before registration
        new_provides = set(getattr(cls, "provides", []))
        if new_provides:
            for existing_name in extractor_registry.list():
                existing_meta = extractor_registry.get_meta(existing_name)
                overlap = new_provides & set(existing_meta.provides)
                if overlap:
                    raise DuplicateRegistrationError(
                        f"Extractor '{name}' provides {overlap} which conflicts "
                        f"with existing extractor '{existing_name}'",
                        "extractor",
                    )
        extractor_registry.register(name, cls=cls, origin="builtin")
        return cls

    return decorator


def register_fusion(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator: register a class as a fusion plugin."""

    def decorator(cls: type[T]) -> type[T]:
        if getattr(cls, "name", None) != name:
            raise RegistryError(
                f"Alias '{name}' != cls.name '{getattr(cls, 'name', None)}'",
                "fusion",
            )
        fusion_registry.register(name, cls=cls, origin="builtin")
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


def scan_directory(registry: Registry[Any], package_path: str) -> None:
    """Import all .py modules in a package directory, triggering decorators.

    Args:
        registry: Target registry (used only for logging context).
        package_path: Dotted package path (e.g., 'video_eval.evaluators').
    """
    try:
        package = importlib.import_module(package_path)
    except ImportError as exc:
        logger.debug(
            "Could not import package '%s' for scan: %s", package_path, exc
        )
        return

    if not hasattr(package, "__path__"):
        return

    for importer, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        if module_name.startswith("_"):
            continue
        full_name = f"{package_path}.{module_name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:
            logger.warning(
                "Failed to import '%s' during scan of '%s' registry: %s",
                full_name,
                registry.name,
                exc,
            )


def discover_entry_points(registry: Registry[Any], group: str) -> None:
    """Discover plugins from installed packages via entry_points.

    Each entry point is registered as a Placeholder (origin='entry-point').
    Errors per EP are caught and logged as warnings without interrupting scan.
    """
    try:
        from importlib.metadata import entry_points as get_entry_points
    except ImportError:
        return

    try:
        eps = get_entry_points(group=group)
    except TypeError:
        # Python < 3.12 compat: entry_points() may not accept group kwarg
        try:
            all_eps = get_entry_points()
            eps = all_eps.get(group, [])  # type: ignore[union-attr]
        except Exception:
            eps = []

    for ep in eps:
        name = ep.name

        def _make_loader(entry_point: Any) -> Callable[[], type]:
            """Create a loader closure that captures the entry point."""
            def loader() -> type:
                return entry_point.load()
            return loader

        placeholder = Placeholder(
            name=name,
            loader=_make_loader(ep),
            origin="entry-point",
            module_path=str(ep.value) if hasattr(ep, "value") else None,
        )

        try:
            registry.register(name, cls=placeholder, origin="entry-point")  # type: ignore[arg-type]
        except DuplicateRegistrationError:
            logger.debug(
                "Entry point '%s' in group '%s' conflicts with existing "
                "registration; skipping.",
                name,
                group,
            )
        except Exception as exc:
            logger.warning(
                "Failed to register entry point '%s' in group '%s': %s",
                name,
                group,
                exc,
            )


# ---------------------------------------------------------------------------
# Global registry instances
# ---------------------------------------------------------------------------
# Created with base_cls=None to avoid circular imports.
# base_cls is set later by initialize_registries() after base classes load.

evaluator_registry: Registry[Any] = Registry("evaluator", base_cls=None)
extractor_registry: Registry[Any] = Registry("extractor", base_cls=None)
backend_registry: Registry[Any] = Registry("backend", base_cls=None)
fusion_registry: Registry[Any] = Registry("fusion", base_cls=None)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_registries() -> None:
    """Initialize all registries: set base_cls, scan builtins, discover EPs.

    Called once at startup. Does NOT freeze registries.
    """
    # Deferred import of base classes to avoid circular dependency.
    # base.py imports decorators from registry.py; registry.py must not
    # import base.py at module level.
    try:
        from video_eval.core.base import (
            BaseBackend,
            BaseEvaluator,
            BaseExtractor,
            BaseFusion,
        )

        evaluator_registry.base_cls = BaseEvaluator
        extractor_registry.base_cls = BaseExtractor
        backend_registry.base_cls = BaseBackend
        fusion_registry.base_cls = BaseFusion
    except ImportError:
        logger.debug("Base classes not available; skipping base_cls assignment.")

    # Scan builtin plugin directories
    scan_directory(evaluator_registry, "video_eval.evaluators")
    scan_directory(extractor_registry, "video_eval.extractors")
    scan_directory(backend_registry, "video_eval.backends")
    scan_directory(fusion_registry, "video_eval.fusions")

    # Discover entry-point plugins
    discover_entry_points(evaluator_registry, "video_eval.evaluators")
    discover_entry_points(extractor_registry, "video_eval.extractors")
    discover_entry_points(backend_registry, "video_eval.backends")
    discover_entry_points(fusion_registry, "video_eval.fusions")
