"""Optimizer and Scheduler Registry for Segmentation Task.

Usage:
    from src.segmentation.registries.optimizers import (
        build_optimizer, build_scheduler, list_optimizers, list_schedulers
    )

    # Build optimizer (requires model parameters as first arg)
    optimizer = build_optimizer("adamw", model.parameters(), lr=1e-3)

    # Build scheduler (requires optimizer as first arg)
    scheduler = build_scheduler("cosine", optimizer, T_max=100)
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from src.common.registry_base import BaseRegistry

# ==============================
# Optimizer Registry
# ==============================

_optimizer_registry = BaseRegistry(name="optimizer")


@_optimizer_registry.register("sgd", aliases=["SGD"])
def build_sgd(params: Iterator, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0, **kwargs):
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


@_optimizer_registry.register("adam", aliases=["Adam"])
def build_adam(params: Iterator, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs):
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


@_optimizer_registry.register("adamw", aliases=["AdamW"])
def build_adamw(params: Iterator, lr: float = 1e-3, weight_decay: float = 1e-2, **kwargs):
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


@_optimizer_registry.register("radam", aliases=["RAdam"])
def build_radam(params: Iterator, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs):
    return optim.RAdam(params, lr=lr, weight_decay=weight_decay)


def _filter_optimizer_kwargs(factory: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs for optimizer factory (exclude 'params')."""
    sig = inspect.signature(factory)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters.keys()) - {"params"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    unknown = sorted(set(kwargs) - set(filtered) - {"type"})

    if unknown:
        warnings.warn(
            f"Unknown parameter(s) for {factory.__name__}: {unknown}. These were ignored.",
            category=UserWarning,
            stacklevel=3,
        )
    return filtered


def build_optimizer(params: Iterator, name: str | None = None, **kwargs: Any) -> optim.Optimizer:
    """Build an optimizer by name.

    The name can be given positionally or via a ``type`` key, so a whole config
    block can be passed directly: ``build_optimizer(model.parameters(), **config["optimizer"])``.

    Args:
        params: Model parameters (required)
        name: Optimizer name (case-insensitive). If omitted, taken from ``type``.
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    if name is None:
        name = kwargs.pop("type", None)
    if name is None:
        raise KeyError("build_optimizer requires a name or a 'type' key.")
    key = name.strip().lower()
    # Trigger lazy import (if any) before reading the registry so that
    # lazily-registered optimizers are resolvable, mirroring BaseRegistry.build().
    _optimizer_registry._ensure_imported(key)
    with _optimizer_registry._lock:
        factory = _optimizer_registry._registry.get(key)

    if factory is None:
        known = list_optimizers()
        raise KeyError(f"Unknown optimizer '{name}'. Known: {known}")

    filtered_kwargs = _filter_optimizer_kwargs(factory, kwargs)
    return factory(params, **filtered_kwargs)


def list_optimizers(prefix: str | None = None) -> list[str]:
    """List registered optimizer names."""
    return _optimizer_registry.list_registered(prefix)


register_optimizer = _optimizer_registry.register

# ==============================
# Scheduler Registry
# ==============================

_scheduler_registry = BaseRegistry(name="scheduler")


@_scheduler_registry.register("cosine", aliases=["CosineAnnealingLR", "cosine_annealing"])
def build_cosine(optimizer: optim.Optimizer, T_max: int = 100, eta_min: float = 0.0, **kwargs):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


@_scheduler_registry.register("step", aliases=["StepLR", "step_lr"])
def build_step(optimizer: optim.Optimizer, step_size: int = 30, gamma: float = 0.1, **kwargs):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


@_scheduler_registry.register("multistep", aliases=["MultiStepLR", "multi_step"])
def build_multistep(optimizer: optim.Optimizer, milestones: list[int] | None = None, gamma: float = 0.1, **kwargs):
    milestones = milestones or [30, 60, 90]
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


@_scheduler_registry.register("exponential", aliases=["ExponentialLR", "exp"])
def build_exponential(optimizer: optim.Optimizer, gamma: float = 0.95, **kwargs):
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


@_scheduler_registry.register("plateau", aliases=["ReduceLROnPlateau", "reduce_on_plateau"])
def build_plateau(optimizer: optim.Optimizer, mode: str = "min", factor: float = 0.1, patience: int = 10, **kwargs):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)


@_scheduler_registry.register("onecycle", aliases=["OneCycleLR", "one_cycle"])
def build_onecycle(
    optimizer: optim.Optimizer,
    max_lr: float = 0.1,
    total_steps: int = 1000,
    pct_start: float = 0.3,
    **kwargs,
):
    return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start)


def _filter_scheduler_kwargs(factory: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs for scheduler factory (exclude 'optimizer')."""
    sig = inspect.signature(factory)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters.keys()) - {"optimizer"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    unknown = sorted(set(kwargs) - set(filtered) - {"type"})

    if unknown:
        warnings.warn(
            f"Unknown parameter(s) for {factory.__name__}: {unknown}. These were ignored.",
            category=UserWarning,
            stacklevel=3,
        )
    return filtered


def build_scheduler(optimizer: optim.Optimizer, name: str | None = None, **kwargs: Any) -> LRScheduler:
    """Build a learning rate scheduler by name.

    The name can be given positionally or via a ``type`` key, so a whole config
    block can be passed directly: ``build_scheduler(optimizer, **config["scheduler"])``.

    Args:
        optimizer: Optimizer instance (required)
        name: Scheduler name (case-insensitive). If omitted, taken from ``type``.
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance
    """
    if name is None:
        name = kwargs.pop("type", None)
    if name is None:
        raise KeyError("build_scheduler requires a name or a 'type' key.")
    key = name.strip().lower()
    # Trigger lazy import (if any) before reading the registry so that
    # lazily-registered schedulers are resolvable, mirroring BaseRegistry.build().
    _scheduler_registry._ensure_imported(key)
    with _scheduler_registry._lock:
        factory = _scheduler_registry._registry.get(key)

    if factory is None:
        known = list_schedulers()
        raise KeyError(f"Unknown scheduler '{name}'. Known: {known}")

    filtered_kwargs = _filter_scheduler_kwargs(factory, kwargs)
    return factory(optimizer, **filtered_kwargs)


def list_schedulers(prefix: str | None = None) -> list[str]:
    """List registered scheduler names."""
    return _scheduler_registry.list_registered(prefix)


register_scheduler = _scheduler_registry.register
