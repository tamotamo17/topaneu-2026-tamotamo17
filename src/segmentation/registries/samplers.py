"""Sampler Registry for Segmentation Task.

Usage:
    from src.segmentation.registries.samplers import build_sampler, list_samplers, register_sampler

    # Build a sampler (requires dataset)
    sampler = build_sampler("random", dataset=dataset)

    # Register a new sampler
    @register_sampler("my_sampler")
    def build_my_sampler(dataset, **kwargs):
        return MySampler(dataset, **kwargs)
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset, RandomSampler, Sampler, SequentialSampler

from src.common.registry_base import BaseRegistry

_registry = BaseRegistry(name="sampler")


@_registry.register("random", aliases=["random_sampler"])
def build_random_sampler(
    dataset: Dataset,
    replacement: bool = False,
    num_samples: int | None = None,
    **kwargs: Any,
) -> RandomSampler:
    """Build a random sampler.

    Args:
        dataset: Dataset to sample from
        replacement: Sample with replacement
        num_samples: Number of samples to draw (default: len(dataset))

    Returns:
        RandomSampler instance
    """
    return RandomSampler(dataset, replacement=replacement, num_samples=num_samples)


@_registry.register("sequential", aliases=["sequential_sampler"])
def build_sequential_sampler(
    dataset: Dataset,
    **kwargs: Any,
) -> SequentialSampler:
    """Build a sequential sampler.

    Args:
        dataset: Dataset to sample from

    Returns:
        SequentialSampler instance
    """
    return SequentialSampler(dataset)


def build_sampler(dataset: Dataset, name: str | None = None, **kwargs: Any) -> Sampler:
    """Build a sampler by name.

    The name can be given positionally or via a ``type`` key, so a whole config
    block can be passed directly: ``build_sampler(dataset, **config["sampler"])``.

    Args:
        dataset: Dataset to sample from
        name: Sampler name (case-insensitive). If omitted, taken from ``type``.
        **kwargs: Additional sampler arguments

    Returns:
        Sampler instance
    """
    if name is None:
        name = kwargs.pop("type", None)
    if name is None:
        raise KeyError("build_sampler requires a name or a 'type' key.")
    return _registry.build(name, dataset=dataset, **kwargs)


def list_samplers(prefix: str | None = None) -> list[str]:
    """List registered sampler names."""
    return _registry.list_registered(prefix)


register_sampler = _registry.register
register_lazy_sampler = _registry.register_lazy
