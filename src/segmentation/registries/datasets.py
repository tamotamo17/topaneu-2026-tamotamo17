"""Dataset Registry for Segmentation Task.

Usage:
    from src.segmentation.registries.datasets import build_dataset, list_datasets, register_dataset

    # Build a dataset
    dataset = build_dataset("random_segmentation", num_samples=1000, num_classes=6)

    # Register a new dataset
    @register_dataset("my_dataset")
    def build_my_dataset(**kwargs):
        return MyDataset(**kwargs)
"""

from src.common.registry_base import BaseRegistry

_registry = BaseRegistry(
    name="dataset",
    lazy_map={
        "random_segmentation": "src.segmentation.dataloaders.datasets",
        "random_mask": "src.segmentation.dataloaders.datasets",
        "dummy_segmentation": "src.segmentation.dataloaders.datasets",
        "fold_segmentation": "src.segmentation.dataloaders.datasets",
        "fold_seg": "src.segmentation.dataloaders.datasets",
        "cv_segmentation": "src.segmentation.dataloaders.datasets",
        "image_mask": "src.segmentation.dataloaders.datasets",
        # 2.5D multiclass dataset (Step2 vessels / Step3 aneurysm-location),
        # with optional vessel-mask conditioning channels (aux_dir).
        "slices_2_5d_mc": "src.segmentation.dataloaders.slices_2_5d_multiclass",
        "seg_2_5d_mc": "src.segmentation.dataloaders.slices_2_5d_multiclass",
        "slice25d_mc": "src.segmentation.dataloaders.slices_2_5d_multiclass",
        # 3D patch dataset (patch-based 3D Step2)
        "patches_3d": "src.segmentation.dataloaders.patches_3d",
        "patch3d": "src.segmentation.dataloaders.patches_3d",
        "seg_3d": "src.segmentation.dataloaders.patches_3d",
        # Step4 fine aneurysm-shape seg (crops)
        "fineseg_crops": "src.segmentation.dataloaders.fineseg_crops",
        "aneurysm_fineseg": "src.segmentation.dataloaders.fineseg_crops",
    },
)

register_dataset = _registry.register
register_lazy_dataset = _registry.register_lazy
build_dataset = _registry.build
list_datasets = _registry.list_registered
