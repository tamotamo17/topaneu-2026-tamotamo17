"""Base Registry Class for Component Management.

This module provides a thread-safe, lazy-loading registry pattern used by all
tasks (classification / segmentation / detection) to register and build
components by name from YAML config.

Usage:
    # Create a registry instance
    _registry = BaseRegistry(name="model", lazy_map={
        "timm_v1": "src.classification.models.timm_models",
    })

    # Use as module-level functions
    register_model = _registry.register
    build_model = _registry.build
    list_models = _registry.list_registered
"""

from __future__ import annotations

import importlib
import inspect
import threading
import warnings
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Thread-safe registry with lazy loading support.

    Features:
        - Thread-safe registration and lookup
        - Lazy module imports (modules are imported on first use)
        - Case-insensitive name lookup
        - Alias support for multiple names mapping to the same factory
        - Automatic kwargs filtering based on factory signature

    Args:
        name: Registry name (used in error messages)
        lazy_map: Initial lazy load map {name: module_path}
    """

    def __init__(self, name: str, lazy_map: dict[str, str] | None = None):
        self._name = name
        self._lock = threading.Lock()
        self._registry: dict[str, Callable[..., T]] = {}
        self._lazy_map: dict[str, str] = lazy_map.copy() if lazy_map else {}

    def _norm(self, name: str) -> str:
        """Normalize name to lowercase for case-insensitive lookup."""
        return name.strip().lower()

    def register(
        self,
        name: str,
        *,
        aliases: list[str] | None = None,
        override: bool = False,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to register a factory function.

        Example:
            @register("simple_cnn", aliases=["cnn", "simplecnn"])
            def build_simple_cnn(**kwargs) -> nn.Module:
                return SimpleCNN(**kwargs)

        Args:
            name: Primary registry key (case-insensitive)
            aliases: Additional names that map to the same factory
            override: If True, allow re-registering an existing name

        Returns:
            Decorator function
        """

        def _decorator(factory: Callable[..., T]) -> Callable[..., T]:
            keys = [self._norm(name)] + ([self._norm(a) for a in aliases] if aliases else [])
            with self._lock:
                for k in keys:
                    if not override and k in self._registry:
                        raise ValueError(f"{self._name} '{k}' is already registered")
                for k in keys:
                    self._registry[k] = factory
                    self._lazy_map.pop(k, None)
            return factory

        return _decorator

    def register_lazy(
        self,
        name: str,
        module_path: str,
        *,
        aliases: list[str] | None = None,
        override: bool = False,
    ) -> None:
        """Register a lazy mapping (name -> module path) without importing.

        This allows build() to import the module on demand, reducing startup time.

        Args:
            name: Primary registry key
            module_path: Python module path (e.g., "src.classification.models.simple_cnn")
            aliases: Additional names for the same mapping
            override: If True, allow overwriting existing registration
        """
        keys = [self._norm(name)] + ([self._norm(a) for a in aliases] if aliases else [])
        with self._lock:
            for k in keys:
                if not override and (k in self._registry or k in self._lazy_map):
                    raise ValueError(f"{self._name} '{k}' is already registered (eager or lazy)")
            for k in keys:
                self._lazy_map[k] = module_path

    def _ensure_imported(self, name: str) -> None:
        """Import the module for `name` if it's registered lazily and not yet loaded."""
        key = self._norm(name)
        with self._lock:
            if key in self._registry:
                return
            mod_path = self._lazy_map.get(key)
        if mod_path:
            # Import outside lock (double import is harmless)
            importlib.import_module(mod_path)

    def _filter_kwargs(
        self,
        factory: Callable[..., Any],
        kwargs: dict[str, Any],
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Filter kwargs to match factory signature.

        If the factory accepts **kwargs, all parameters are passed through.
        Otherwise, only parameters in the factory's signature are kept.

        Args:
            factory: The factory function to inspect
            kwargs: Input keyword arguments
            strict: If True, raise TypeError for unknown parameters

        Returns:
            Filtered kwargs dict
        """
        sig = inspect.signature(factory)

        # If factory accepts **kwargs, no filtering needed
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs

        allowed = sig.parameters.keys()
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        unknown = sorted(set(kwargs) - set(filtered))

        if unknown:
            msg = f"Unknown parameter(s) for {factory.__name__}: {unknown}. Allowed: {sorted(allowed)}. These were ignored."
            if strict:
                raise TypeError(msg)
            warnings.warn(msg, category=UserWarning, stacklevel=4)

        return filtered

    def build(self, name: str | None = None, **kwargs: Any) -> T:
        """Build an instance by name.

        The name can be given positionally, or as a ``type`` key inside kwargs.
        The latter lets callers pass a whole config block directly, e.g.
        ``build(**config["model"])`` where ``config["model"]`` includes ``type``.

        Args:
            name: Registered name (case-insensitive). If omitted, taken from the
                ``type`` keyword argument.
            **kwargs: Arguments passed to the factory function (``type`` is consumed
                here and not forwarded to the factory).

        Returns:
            Instance created by the factory

        Raises:
            KeyError: If name is not provided or not registered
        """
        if name is None:
            name = kwargs.pop("type", None)
        if name is None:
            raise KeyError(f"{self._name} build requires a name (positional) or a 'type' key.")
        key = self._norm(name)
        self._ensure_imported(key)

        with self._lock:
            factory = self._registry.get(key)

        if factory is None:
            # Double-check after potential concurrent import
            self._ensure_imported(key)
            with self._lock:
                factory = self._registry.get(key)
            if factory is None:
                with self._lock:
                    known = sorted(set(list(self._registry.keys()) + list(self._lazy_map.keys())))
                raise KeyError(f"Unknown {self._name} '{name}'. Known: [{', '.join(known)}]")

        params = self._filter_kwargs(factory, kwargs)
        return factory(**params)

    def list_registered(self, prefix: str | None = None) -> list[str]:
        """List registered names (sorted).

        Includes both eagerly registered and lazily mapped names.

        Args:
            prefix: Optional prefix filter (case-insensitive)

        Returns:
            Sorted list of registered names
        """
        with self._lock:
            names = sorted(set(list(self._registry.keys()) + list(self._lazy_map.keys())))
        if prefix is not None:
            p = self._norm(prefix)
            names = [n for n in names if n.startswith(p)]
        return names

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered."""
        key = self._norm(name)
        with self._lock:
            return key in self._registry or key in self._lazy_map
