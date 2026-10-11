"""Segmentation Models.

Available models are registered via the model registry.
See src/segmentation/registries/models.py for the full list.

Registered models:
- model_v1 (aliases: segmentation_v1, smp_v1): Unified SMP builder.
  Select architecture/backbone via the `arch` and `backbone` arguments,
  e.g. build_model("model_v1", arch="Unet", backbone="resnet50", num_classes=6).
  Supported `arch`: Unet, FPN, DeepLabV3Plus, PSPNet, ... (any smp architecture).
  Supported `backbone`: resnet50, efficientnet-b3, ... (any smp encoder).
"""
