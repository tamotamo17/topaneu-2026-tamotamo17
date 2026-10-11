"""Model Registry for Detection Task.

Usage:
    from src.detection.registries.models import build_model, list_models, register_model

    # List available models
    print(list_models())

    # Build a model
    model = build_model("rfdetr_base", num_classes=6, pretrained=False)

    # Register a new model
    @register_model("my_detector", aliases=["mydet"])
    def build_my_detector(**kwargs):
        return MyDetectorWrapper(**kwargs)
"""

from src.common.registry_base import BaseRegistry

# Create registry instance with lazy-load mappings
_registry = BaseRegistry(
    name="model",
    lazy_map={
        "rfdetr_nano": "src.detection.models.rfdetr_wrapper",
        "rfdetr_small": "src.detection.models.rfdetr_wrapper",
        "rfdetr_medium": "src.detection.models.rfdetr_wrapper",
        "rfdetr_base": "src.detection.models.rfdetr_wrapper",
        "rfdetr": "src.detection.models.rfdetr_wrapper",
        "rfdetr_large": "src.detection.models.rfdetr_wrapper",
    },
)

register_model = _registry.register
register_lazy_model = _registry.register_lazy
build_model = _registry.build
list_models = _registry.list_registered
