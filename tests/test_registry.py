"""Unit tests for the registry system (BaseRegistry and the task registries).

All models are built with encoder_weights=None / pretrained=False so the tests
run offline.
"""

import pytest
import torch


def _build_auxbin(num_classes: int = 4, in_channels: int = 5):
    from src.segmentation.registries.models import build_model

    return build_model(
        "model_v2_auxbin",
        arch="Unet",
        backbone="resnet18",
        encoder_weights=None,
        in_channels=in_channels,
        num_classes=num_classes,
    )


class TestBaseRegistry:
    """Tests for the base registry class."""

    def test_registry_creation(self):
        """A registry can be created with a name."""
        from src.common.registry_base import BaseRegistry

        registry = BaseRegistry(name="test")
        assert registry._name == "test"
        assert registry.list_registered() == []

    def test_register_and_build(self):
        """Basic register -> build flow."""
        from src.common.registry_base import BaseRegistry

        registry = BaseRegistry[int](name="test")

        @registry.register("double")
        def double(x: int) -> int:
            return x * 2

        assert registry.build("double", x=5) == 10

    def test_build_from_type_key(self):
        """The name can be passed as a `type` key (whole config block)."""
        from src.common.registry_base import BaseRegistry

        registry = BaseRegistry[int](name="test")

        @registry.register("double")
        def double(x: int) -> int:
            return x * 2

        assert registry.build(**{"type": "double", "x": 4}) == 8

    def test_case_insensitive_lookup(self):
        """Lookup is case-insensitive."""
        from src.common.registry_base import BaseRegistry

        registry = BaseRegistry[int](name="test")

        @registry.register("Double")
        def double(x: int) -> int:
            return x * 2

        assert registry.build("DOUBLE", x=1) == 2
        assert registry.is_registered("double")

    def test_aliases(self):
        """All aliases resolve to the same factory."""
        from src.common.registry_base import BaseRegistry

        registry = BaseRegistry[str](name="test")

        @registry.register("hello", aliases=["hi", "hey"])
        def greet(person_name: str) -> str:
            return f"Hello, {person_name}!"

        for name in ("hello", "hi", "hey"):
            assert registry.build(name, person_name="World") == "Hello, World!"

    def test_duplicate_registration_raises(self):
        """Re-registering a name raises unless override=True."""
        from src.common.registry_base import BaseRegistry

        registry = BaseRegistry[int](name="test")

        @registry.register("one")
        def one() -> int:
            return 1

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("one")
            def one_again() -> int:
                return 1

        @registry.register("one", override=True)
        def one_override() -> int:
            return 11

        assert registry.build("one") == 11

    def test_unknown_name_raises(self):
        """Building an unknown name raises KeyError."""
        from src.common.registry_base import BaseRegistry

        registry = BaseRegistry(name="test")
        with pytest.raises(KeyError, match="Unknown test"):
            registry.build("missing")

    def test_unknown_kwargs_are_filtered(self):
        """Unknown kwargs are dropped with a warning for factories without **kwargs."""
        from src.common.registry_base import BaseRegistry

        registry = BaseRegistry[int](name="test")

        @registry.register("double")
        def double(x: int) -> int:
            return x * 2

        with pytest.warns(UserWarning, match="Unknown parameter"):
            assert registry.build("double", x=3, unused=1) == 6

    def test_list_registered(self):
        """Listing includes eager and lazy entries and supports a prefix filter."""
        from src.common.registry_base import BaseRegistry

        registry = BaseRegistry(name="test", lazy_map={"lazy_item": "some.module"})

        @registry.register("item_a")
        def a():
            pass

        @registry.register("item_b")
        def b():
            pass

        items = registry.list_registered()
        assert items == sorted(items)
        assert {"item_a", "item_b", "lazy_item"} <= set(items)
        assert registry.list_registered(prefix="item") == ["item_a", "item_b"]


class TestSegmentationModelRegistry:
    """Tests for the segmentation model registry."""

    def test_list_models(self):
        from src.segmentation.registries.models import list_models

        models = list_models()
        for name in ("model_v1", "model_v2_auxbin", "monai_segresnet3d",
                     "monai_segresnet3d_auxcls", "monai_dynunet3d_auxcls"):
            assert name in models

    def test_build_model_v2_auxbin(self):
        """model_v2_auxbin returns the main masks plus a 1-channel auxiliary vessel mask."""
        model = _build_auxbin(num_classes=4, in_channels=5)
        output = model({"image": torch.randn(2, 5, 64, 64)})
        assert output["masks"].shape == (2, 4, 64, 64)
        assert output["aux_masks"].shape == (2, 1, 64, 64)

    def test_model_aliases(self):
        """Aliases build the same model class."""
        from src.segmentation.registries.models import build_model

        kw = dict(arch="Unet", backbone="resnet18", encoder_weights=None, num_classes=2)
        assert type(build_model("model_v2_auxbin", **kw)) is type(build_model("smp_auxbin", **kw))
        assert type(build_model("model_v1", **kw)) is type(build_model("smp_v1", **kw))

    def test_build_3d_model(self):
        """3D SegResNet follows the {"masks": [B, C, D, H, W]} contract."""
        from src.segmentation.registries.models import build_model

        model = build_model("monai_segresnet3d", in_channels=2, num_classes=1, init_filters=8)
        output = model({"image": torch.randn(1, 2, 32, 32, 32)})
        assert output["masks"].shape == (1, 1, 32, 32, 32)


class TestSegmentationLossRegistry:
    """Tests for the segmentation loss registry."""

    def test_list_losses(self):
        from src.segmentation.registries.losses import list_losses

        losses = list_losses()
        for name in ("dice", "bce_dice", "ce_dice", "focal_tversky_ce", "heatmap_focal"):
            assert name in losses

    def test_build_focal_tversky_ce(self):
        """focal_tversky_ce on multiclass logits with an auxiliary binary head."""
        from src.segmentation.registries.losses import build_loss

        num_classes = 4
        loss_fn = build_loss(
            "focal_tversky_ce",
            alpha=0.2,
            beta=0.8,
            gamma=2.0,
            dice_weight=0.3,
            aux_weight=0.3,
            class_weights=[0.2, 1.0, 1.0, 2.0],
        )
        outputs = {
            "masks": torch.randn(2, num_classes, 32, 32, requires_grad=True),
            "aux_masks": torch.randn(2, 1, 32, 32, requires_grad=True),
        }
        targets = {"masks": torch.randint(0, num_classes, (2, 1, 32, 32))}
        loss = loss_fn(outputs, targets)
        assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
        loss.backward()
        assert outputs["aux_masks"].grad is not None

    def test_build_heatmap_focal(self):
        """heatmap_focal on a 3D peaked target."""
        from src.segmentation.registries.losses import build_loss

        loss_fn = build_loss(type="heatmap_focal", gamma=2.0, fg_weight=50.0, normalize="mean")
        logits = torch.randn(2, 1, 16, 16, 16)
        target = torch.zeros(2, 1, 16, 16, 16)
        target[0, 0, 8, 8, 8] = 1.0
        loss = loss_fn(logits, target)
        assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0

    def test_heatmap_focal_rejects_bad_normalize(self):
        from src.segmentation.registries.losses import build_loss

        with pytest.raises(ValueError, match="normalize"):
            build_loss("heatmap_focal", normalize="bogus")

    def test_builtin_fallback(self):
        """Unregistered names fall back to torch.nn built-in losses."""
        from src.segmentation.registries.losses import build_loss

        assert isinstance(build_loss("CrossEntropyLoss"), torch.nn.CrossEntropyLoss)

    def test_unknown_loss_raises(self):
        from src.segmentation.registries.losses import build_loss

        with pytest.raises(KeyError, match="Unknown loss"):
            build_loss("definitely_not_a_loss")


class TestSegmentationDatasetRegistry:
    """Tests for the segmentation dataset registry."""

    def test_list_datasets(self):
        from src.segmentation.registries.datasets import list_datasets

        datasets = list_datasets()
        for name in ("random_segmentation", "slices_2_5d_mc", "patches_3d", "fineseg_crops"):
            assert name in datasets

    def test_build_random_dataset(self):
        from src.segmentation.registries.datasets import build_dataset

        dataset = build_dataset("random_segmentation", num_samples=10, image_size=32, num_classes=3)
        assert len(dataset) == 10
        sample = dataset[0]
        assert sample["image"].shape == (3, 32, 32)
        assert sample["masks"].shape == (3, 32, 32)


class TestSegmentationOptimizerRegistry:
    """Tests for the optimizer / scheduler registries."""

    def test_list_optimizers(self):
        from src.segmentation.registries.optimizers import list_optimizers

        optimizers = list_optimizers()
        assert "adamw" in optimizers
        assert "sgd" in optimizers

    def test_build_adamw(self):
        from src.segmentation.registries.optimizers import build_optimizer

        model = torch.nn.Linear(10, 5)
        optimizer = build_optimizer(model.parameters(), name="adamw", lr=1e-3, weight_decay=1e-4)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["weight_decay"] == 1e-4

    def test_build_optimizer_from_config_block(self):
        """A whole config block (with `type`) can be passed directly."""
        from src.segmentation.registries.optimizers import build_optimizer

        model = torch.nn.Linear(10, 5)
        optimizer = build_optimizer(model.parameters(), **{"type": "adamw", "lr": 1e-3})
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_list_schedulers(self):
        from src.segmentation.registries.optimizers import list_schedulers

        schedulers = list_schedulers()
        assert "cosine" in schedulers
        assert "step" in schedulers

    def test_build_cosine_scheduler(self):
        from src.segmentation.registries.optimizers import build_optimizer, build_scheduler

        model = torch.nn.Linear(10, 5)
        optimizer = build_optimizer(model.parameters(), name="adamw", lr=1e-3)
        scheduler = build_scheduler(optimizer, name="cosine", T_max=90)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 90


class TestSegmentationSamplerRegistry:
    """Tests for the sampler registry."""

    def test_list_samplers(self):
        from src.segmentation.registries.samplers import list_samplers

        samplers = list_samplers()
        assert "random" in samplers
        assert "sequential" in samplers

    def test_build_samplers(self):
        from torch.utils.data import RandomSampler, SequentialSampler

        from src.segmentation.registries.datasets import build_dataset
        from src.segmentation.registries.samplers import build_sampler

        dataset = build_dataset("random_segmentation", num_samples=8, image_size=16, num_classes=2)
        assert isinstance(build_sampler(dataset, name="random"), RandomSampler)
        sequential = build_sampler(dataset, **{"type": "sequential"})
        assert isinstance(sequential, SequentialSampler)
        assert list(sequential) == list(range(8))


class TestSegmentationLightningModuleRegistry:
    """Tests for the LightningModule registry."""

    def test_list_lightning_modules(self):
        from src.segmentation.registries.lightning_modules import list_lightning_modules

        modules = list_lightning_modules()
        for name in ("segmentation", "segmentation_vol_mc", "segmentation_3d", "heatmap_3d"):
            assert name in modules

    def test_build_volumetric_multiclass_module(self):
        """segmentation_vol_mc wraps model_v2_auxbin + focal_tversky_ce."""
        from src.segmentation.registries.lightning_modules import build_lightning_module
        from src.segmentation.registries.losses import build_loss
        from src.segmentation.registries.optimizers import build_optimizer

        model = _build_auxbin(num_classes=4, in_channels=5)
        module = build_lightning_module(
            "segmentation_vol_mc",
            model=model,
            loss=build_loss("focal_tversky_ce", aux_weight=0.3),
            optimizer=build_optimizer(model.parameters(), name="adamw", lr=1e-3),
            num_classes=4,
        )
        output = module({"image": torch.randn(2, 5, 32, 32)})
        assert output["masks"].shape == (2, 4, 32, 32)


class TestDetectionRegistry:
    """Tests for the detection registries (RF-DETR itself is not built)."""

    def test_list_models(self):
        from src.detection.registries.models import list_models

        models = list_models()
        for name in ("rfdetr_nano", "rfdetr_small", "rfdetr_medium", "rfdetr_base", "rfdetr_large"):
            assert name in models

    def test_list_datasets(self):
        from src.detection.registries.datasets import list_datasets

        datasets = list_datasets()
        assert "random_detection" in datasets
        assert "coco_detection" in datasets

    def test_only_models_and_datasets_registries(self):
        """Detection intentionally exposes only the models and datasets registries."""
        import src.detection.registries as det

        assert sorted(det.__all__) == ["datasets", "models"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
