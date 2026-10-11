"""Lightning Module Registry for Segmentation Task.

Usage:
    from src.segmentation.registries.lightning_modules import (
        build_lightning_module, list_lightning_modules, register_lightning_module
    )

    # Build a lightning module
    module = build_lightning_module(
        "segmentation",
        model=model,
        loss=loss,
        optimizer=optimizer,
        num_classes=6,
    )

    # Register a new module
    @register_lightning_module("my_module")
    def build_my_module(**kwargs):
        return MyModule(**kwargs)
"""

from src.common.registry_base import BaseRegistry

_registry = BaseRegistry(
    name="lightning_module",
    lazy_map={
        "segmentation": "src.segmentation.lightning_module.module",
        "seg": "src.segmentation.lightning_module.module",
        "segment": "src.segmentation.lightning_module.module",
        # Multiclass (softmax) per-case 3D Dice eval — Step2 / Step3
        "segmentation_vol_mc": "src.segmentation.lightning_module.volumetric_multiclass_module",
        "seg_vol_mc": "src.segmentation.lightning_module.volumetric_multiclass_module",
        "volumetric_multiclass_segmentation": "src.segmentation.lightning_module.volumetric_multiclass_module",
        # True 3D: patch-train / sliding-window-val
        "segmentation_3d": "src.segmentation.lightning_module.volumetric_3d_module",
        "seg_3d_vol": "src.segmentation.lightning_module.volumetric_3d_module",
        "volumetric_3d": "src.segmentation.lightning_module.volumetric_3d_module",
        # Gaussian-heatmap aneurysm detector (Step3)
        "heatmap_3d": "src.segmentation.lightning_module.heatmap_3d_module",
        "gaussian_heatmap_3d": "src.segmentation.lightning_module.heatmap_3d_module",
    },
)

register_lightning_module = _registry.register
register_lazy_lightning_module = _registry.register_lazy
build_lightning_module = _registry.build
list_lightning_modules = _registry.list_registered
