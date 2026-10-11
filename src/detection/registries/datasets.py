"""Dataset Registry for Detection Task.

Usage:
    from src.detection.registries.datasets import build_dataset, list_datasets, register_dataset

    # Build a dataset
    dataset = build_dataset("random_detection", num_samples=100, num_classes=6)
    dataset = build_dataset("coco_detection", coco_root="./data", split="train")

    # Register a new dataset
    @register_dataset("my_dataset")
    def build_my_dataset(**kwargs):
        return MyDataset(**kwargs)
"""

from src.common.registry_base import BaseRegistry

_registry = BaseRegistry(
    name="dataset",
    lazy_map={
        "random_detection": "src.detection.dataloaders.datasets",
        "random": "src.detection.dataloaders.datasets",
        "dummy": "src.detection.dataloaders.datasets",
        "coco_detection": "src.detection.dataloaders.datasets",
        "coco": "src.detection.dataloaders.datasets",
    },
)

register_dataset = _registry.register
register_lazy_dataset = _registry.register_lazy
build_dataset = _registry.build
list_datasets = _registry.list_registered
