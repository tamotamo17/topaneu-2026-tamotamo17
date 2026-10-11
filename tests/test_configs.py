"""Smoke tests for the YAML configs under configs/ (configs/_orig/ is excluded).

Each config is loaded with OmegaConf. Segmentation configs build their model
(offline: encoder_weights=None / pretrained=False, init_from_ckpt removed) and
loss, and must reference a registered lightning_module. Detection configs only
check that model.type is registered (RF-DETR is not built).
"""

import inspect
from pathlib import Path

import pytest

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"
EXCLUDED_DIRS = {"_orig"}


def _collect_configs() -> list[Path]:
    if not CONFIG_ROOT.is_dir():
        return []
    paths = sorted(CONFIG_ROOT.rglob("*.yaml")) + sorted(CONFIG_ROOT.rglob("*.yml"))
    return [p for p in paths if not EXCLUDED_DIRS & set(p.relative_to(CONFIG_ROOT).parts[:-1])]


CONFIGS = _collect_configs()


def _task_of(path: Path) -> str:
    return "detection" if "detection" in path.relative_to(CONFIG_ROOT).parts[:-1] else "segmentation"


def _load(path: Path) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict), f"{path} does not contain a mapping"
    return container


def _offline_model_kwargs(registry, model_cfg: dict) -> dict:
    """Strip checkpoint warm-starts and disable pretrained-weight downloads."""
    kwargs = dict(model_cfg)
    kwargs.pop("init_from_ckpt", None)

    name = kwargs["type"]
    registry._ensure_imported(name)
    factory = registry._registry.get(registry._norm(name))
    params = inspect.signature(factory).parameters if factory is not None else {}
    if "encoder_weights" in kwargs or "encoder_weights" in params:
        kwargs["encoder_weights"] = None
    if "pretrained" in kwargs or "pretrained" in params:
        kwargs["pretrained"] = False
    return kwargs


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(CONFIG_ROOT)) for p in paths]


@pytest.mark.parametrize("path", CONFIGS or [None], ids=_ids(CONFIGS) or ["no-configs"])
class TestConfigs:
    """Every config loads and its components resolve through the task registries."""

    def test_loads(self, path):
        if path is None:
            pytest.skip("no YAML configs under configs/ (excluding configs/_orig/)")
        assert _load(path)

    def test_components_resolve(self, path):
        if path is None:
            pytest.skip("no YAML configs under configs/ (excluding configs/_orig/)")
        cfg = _load(path)
        if "model" not in cfg:
            pytest.skip(f"{path.name} has no model section")
        assert "type" in cfg["model"], f"{path.name}: model.type is missing"

        if _task_of(path) == "detection":
            from src.detection.registries import models as det_models

            assert det_models._registry.is_registered(cfg["model"]["type"])
            return

        from src.segmentation.registries import lightning_modules, losses, models

        model = models.build_model(**_offline_model_kwargs(models._registry, cfg["model"]))
        assert model is not None

        if "loss" in cfg:
            assert losses.build_loss(**cfg["loss"]) is not None

        if "lightning_module" in cfg:
            lm_type = cfg["lightning_module"]["type"]
            assert lightning_modules._registry.is_registered(lm_type), (
                f"{path.name}: lightning_module.type {lm_type!r} is not registered"
            )
