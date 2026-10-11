"""Model Registry for Segmentation Task.

Usage:
    from src.segmentation.registries.models import build_model, list_models, register_model

    # List available models
    print(list_models())

    # Build a model
    model = build_model("model_v1", arch="Unet", backbone="resnet50", num_classes=6)

    # Register a new model
    @register_model("my_model", aliases=["mymodel"])
    def build_my_model(**kwargs):
        return MyModel(**kwargs)
"""

from src.common.registry_base import BaseRegistry

# Create registry instance with lazy-load mappings for SMP models
_registry = BaseRegistry(
    name="model",
    lazy_map={
        # Model v1 (unified interface with arch and backbone arguments)
        "model_v1": "src.segmentation.models.smp_models",
        "segmentation_v1": "src.segmentation.models.smp_models",
        "smp_v1": "src.segmentation.models.smp_models",
        # Model v2: adds an auxiliary binary (vessel/background) segmentation head
        "model_v2_auxbin": "src.segmentation.models.smp_models",
        "smp_auxbin": "src.segmentation.models.smp_models",
        # MONAI 3D networks (patch-based 3D Step2)
        "monai_unet3d": "src.segmentation.models.monai_models",
        "unet3d": "src.segmentation.models.monai_models",
        "monai_unet3d_auxbin": "src.segmentation.models.monai_models",
        "unet3d_auxbin": "src.segmentation.models.monai_models",
        "monai_segresnet3d": "src.segmentation.models.monai_models",
        "segresnet3d": "src.segmentation.models.monai_models",
        "monai_segresnet3d_auxbin": "src.segmentation.models.monai_models",
        "segresnet3d_auxbin": "src.segmentation.models.monai_models",
        # Step3 detector: + auxiliary patch-level aneurysm-presence cls head
        "monai_segresnet3d_auxcls": "src.segmentation.models.monai_models",
        "segresnet3d_auxcls": "src.segmentation.models.monai_models",
        "monai_dynunet3d_auxbin": "src.segmentation.models.monai_models",
        "dynunet3d_auxbin": "src.segmentation.models.monai_models",
        "monai_dynunet3d": "src.segmentation.models.monai_models",
        "dynunet3d": "src.segmentation.models.monai_models",
        "monai_dynunet3d_auxcls": "src.segmentation.models.monai_models",
        "dynunet3d_auxcls": "src.segmentation.models.monai_models",
    },
)

register_model = _registry.register
register_lazy_model = _registry.register_lazy
build_model = _registry.build
list_models = _registry.list_registered
