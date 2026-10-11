"""Tests for the segmentation task (registries, pipeline, and Step3/Step4 components).

Uses encoder_weights=None and small backbones to avoid network access.
"""

import torch


def _build_model(num_classes: int = 3):
    from src.segmentation.registries.models import build_model

    return build_model(
        "model_v1",
        arch="Unet",
        backbone="resnet18",
        encoder_weights=None,
        num_classes=num_classes,
    )


def _make_batch(num_classes: int = 3, batch_size: int = 2, size: int = 64):
    return {
        "image": torch.randn(batch_size, 3, size, size),
        "masks": torch.randint(0, 2, (batch_size, num_classes, size, size)).float(),
    }


class TestSegmentationRegistry:
    """Tests for the segmentation registries."""

    def test_list_models(self):
        from src.segmentation.registries.models import list_models

        models = list_models()
        assert "model_v1" in models

    def test_list_losses(self):
        from src.segmentation.registries.losses import list_losses

        losses = list_losses()
        assert "dice" in losses
        assert "bce_dice" in losses

    def test_list_datasets(self):
        from src.segmentation.registries.datasets import list_datasets

        datasets = list_datasets()
        assert "random_segmentation" in datasets
        assert "fold_segmentation" in datasets

    def test_loss_aliases_resolve(self):
        """Aliases in lazy_map and in register() are in sync."""
        from src.segmentation.registries.losses import build_loss

        for name in ["dice", "diceloss", "bce", "bceloss", "bce_dice", "bcedice", "dice_bce"]:
            assert build_loss(name) is not None


class TestSegmentationPipeline:
    """Integration tests for the segmentation training pipeline."""

    def test_forward_pass(self):
        """Output follows the {"masks": [B, C, H, W]} contract."""
        model = _build_model(num_classes=3)
        output = model({"image": torch.randn(2, 3, 64, 64)})

        assert "masks" in output
        assert output["masks"].shape == (2, 3, 64, 64)

    def test_dice_loss_computation(self):
        from src.segmentation.registries.losses import build_loss

        model = _build_model(num_classes=3)
        loss_fn = build_loss("dice")

        batch = _make_batch(num_classes=3)
        output = model(batch)
        loss = loss_fn(output, batch)

        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_dataset_with_transforms_key(self):
        """Datasets accept the `transforms` (plural) keyword."""
        from src.segmentation.registries.datasets import build_dataset

        dataset = build_dataset(
            "random_segmentation",
            num_samples=8,
            image_size=64,
            num_classes=3,
            transforms=None,
        )
        sample = dataset[0]
        assert "image" in sample
        assert "masks" in sample
        assert sample["masks"].shape[0] == 3

    def test_lightning_module_steps(self):
        from src.segmentation.registries.lightning_modules import build_lightning_module
        from src.segmentation.registries.losses import build_loss
        from src.segmentation.registries.optimizers import build_optimizer

        model = _build_model(num_classes=3)
        loss = build_loss("bce_dice")
        optimizer = build_optimizer(model.parameters(), name="adamw", lr=1e-3)
        module = build_lightning_module("segmentation", model=model, loss=loss, optimizer=optimizer, num_classes=3)

        batch = _make_batch(num_classes=3)

        train_loss = module.training_step(batch, 0)
        assert train_loss is not None and not torch.isnan(train_loss)

        module.eval()
        val_loss = module.validation_step(batch, 0)
        assert val_loss is not None and not torch.isnan(val_loss)


class TestHeatmap3DAuxCls:
    """Auxiliary classification head of the Step3 detector (branched off the encoder bottleneck).

    Adds a "does this patch contain an aneurysm centre?" auxiliary task on top of the
    Gaussian centre-heatmap regression. Also pins that existing configs with
    ``cls_weight=0.0`` behave exactly as before.
    """

    @staticmethod
    def _heatmap_batch(n_pos: int, n_neg: int, size: int = 32):
        """Build a soft target with a 1.0 peak at the centre of positive patches only."""
        b = n_pos + n_neg
        masks = torch.zeros(b, 1, size, size, size)
        for i in range(n_pos):
            masks[i, 0, size // 2, size // 2, size // 2] = 1.0
            masks[i, 0, size // 2 + 1, size // 2, size // 2] = 0.6   # tail
        return {"image": torch.randn(b, 2, size, size, size), "masks": masks}

    @staticmethod
    def _build(cls_pool: str = "avgmax"):
        from src.segmentation.registries.models import build_model

        return build_model("monai_segresnet3d_auxcls", in_channels=2, num_classes=1,
                           init_filters=8, cls_pool=cls_pool)

    def test_seg_path_identical_to_plain_segresnet(self):
        """The seg path is identical to monai_segresnet3d (old checkpoints load as-is)."""
        from src.segmentation.registries.models import build_model

        aux = self._build()
        plain = build_model("monai_segresnet3d", in_channels=2, num_classes=1, init_filters=8)
        plain.net.load_state_dict(aux.net.state_dict())
        aux.eval(), plain.eval()
        batch = self._heatmap_batch(1, 1)
        with torch.no_grad():
            assert torch.equal(aux(batch)["masks"], plain(batch)["masks"])

    def test_cls_head_shape_and_zero_init(self):
        """Zero-initialised output layer => initial logits are 0 => BCE starts at ln2."""
        import math

        model = self._build()
        model.eval()
        with torch.no_grad():
            logits = model(self._heatmap_batch(1, 1))["cls_logits"]
        assert logits.shape == (2, 1)
        assert torch.equal(logits, torch.zeros_like(logits))
        assert math.isclose(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 0], torch.tensor([1.0, 0.0])).item(),
            math.log(2), rel_tol=1e-5)

    def test_pool_variants(self):
        for pool, mult in (("avg", 1), ("max", 1), ("avgmax", 2)):
            model = self._build(cls_pool=pool)
            assert model.cls_head[0].normalized_shape == (64 * mult,)
            assert model(self._heatmap_batch(1, 1))["cls_logits"].shape == (2, 1)

    def test_aux_loss_weighting_and_gradient(self):
        """total = seg + cls_weight * cls, and after one step gradients reach the shared encoder.

        The output layer is zero-initialised, so at step 0 only the last layer gets a
        gradient (backpropagating through W=0 zeroes everything upstream). This is
        intentional: a warm-started encoder should not receive random gradients from
        a freshly initialised head. After one update gradients reach the encoder.
        Both stages are pinned here.
        """
        from src.segmentation.registries.lightning_modules import build_lightning_module
        from src.segmentation.registries.losses import build_loss
        from src.segmentation.registries.optimizers import build_optimizer

        model = self._build()
        optimizer = build_optimizer(model.parameters(), name="adamw", lr=3e-4)
        module = build_lightning_module(
            "heatmap_3d", model=model,
            loss=build_loss(type="heatmap_focal", gamma=2.0, fg_weight=50.0, normalize="mean"),
            optimizer=optimizer,
            roi_size=32, sw_batch_size=2, val_thr=0.5, cls_weight=0.003, cls_thr=0.5)
        logged: dict[str, float] = {}
        module.log_dict = lambda d, **kw: logged.update({k: float(v.detach()) for k, v in d.items()})
        module.log = lambda k, v, **kw: logged.__setitem__(k, float(v.detach() if torch.is_tensor(v) else v))

        module.training_step(self._heatmap_batch(2, 2), 0).backward()
        assert {"train_loss", "train_seg", "train_cls"} <= logged.keys()
        assert abs(logged["train_loss"] - (logged["train_seg"] + 0.003 * logged["train_cls"])) < 1e-6
        assert model.cls_head[-1].weight.grad.abs().sum() > 0     # step 0: last layer only
        assert model.cls_head[2].weight.grad.abs().sum() == 0

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        module.training_step(self._heatmap_batch(2, 2), 1).backward()
        assert model.cls_head[2].weight.grad.abs().sum() > 0      # after one step: whole head
        enc = model.net.down_layers[-1][-1].conv1.conv.weight     # encoder bottleneck
        assert enc.grad is not None and enc.grad.abs().sum() > 0

    def test_cls_target_is_peak_in_patch(self):
        """A patch that only contains the tail stays negative (no firing on partial evidence)."""
        from src.segmentation.registries.lightning_modules import build_lightning_module
        from src.segmentation.registries.losses import build_loss
        from src.segmentation.registries.optimizers import build_optimizer

        model = self._build()
        module = build_lightning_module(
            "heatmap_3d", model=model, loss=build_loss(type="heatmap_focal"),
            optimizer=build_optimizer(model.parameters(), name="adamw", lr=3e-4),
            roi_size=32, cls_weight=0.003, cls_thr=0.5)
        masks = self._heatmap_batch(1, 1)["masks"]
        masks[1, 0, 4, 4, 4] = 0.4                       # tail only (< cls_thr)
        assert (masks[:, 0] > module.cls_thr).flatten(1).any(1).float().tolist() == [1.0, 0.0]

    def test_case_level_auc_and_batch_guard(self):
        """The max sliding-window aux logit is the case score; not recorded for batch > 1 (cases would mix)."""
        from src.segmentation.registries.lightning_modules import build_lightning_module
        from src.segmentation.registries.losses import build_loss
        from src.segmentation.registries.optimizers import build_optimizer

        model = self._build()
        module = build_lightning_module(
            "heatmap_3d", model=model, loss=build_loss(type="heatmap_focal"),
            optimizer=build_optimizer(model.parameters(), name="adamw", lr=3e-4),
            roi_size=32, sw_batch_size=2, overlap=0.5, val_thr=0.5, cls_weight=0.003)
        logged: dict[str, float] = {}
        module.log = lambda k, v, **kw: logged.__setitem__(k, float(v.detach() if torch.is_tensor(v) else v))
        module.eval()

        module.on_validation_epoch_start()
        with torch.no_grad():
            for i in range(4):
                module.validation_step(
                    self._heatmap_batch(1, 0, size=48) if i < 2 else self._heatmap_batch(0, 1, size=48), i)
        assert module._cls_label == [1, 1, 0, 0] and len(module._cls_score) == 4
        module.on_validation_epoch_end()
        assert "val_cls_auc" in logged

        module.on_validation_epoch_start()
        with torch.no_grad():
            module.validation_step(self._heatmap_batch(1, 1, size=48), 0)
        assert module._cls_score == []

    def test_seg_only_model_unchanged_and_guarded(self):
        """cls_weight=0.0 keeps existing behaviour; cls_weight > 0 with a seg-only model raises immediately."""
        import pytest

        from src.segmentation.registries.lightning_modules import build_lightning_module
        from src.segmentation.registries.losses import build_loss
        from src.segmentation.registries.models import build_model
        from src.segmentation.registries.optimizers import build_optimizer

        plain = build_model("monai_segresnet3d", in_channels=2, num_classes=1, init_filters=8)
        loss = build_loss(type="heatmap_focal")
        module = build_lightning_module(
            "heatmap_3d", model=plain, loss=loss,
            optimizer=build_optimizer(plain.parameters(), name="adamw", lr=3e-4),
            roi_size=32, sw_batch_size=2, val_thr=0.5)
        logged: dict[str, float] = {}
        module.log = lambda k, v, **kw: logged.__setitem__(k, float(v.detach() if torch.is_tensor(v) else v))
        module.training_step(self._heatmap_batch(2, 2), 0)
        module.on_validation_epoch_start()
        module.eval()
        with torch.no_grad():
            module.validation_step(self._heatmap_batch(1, 0, size=48), 0)
        module.on_validation_epoch_end()
        assert sorted(logged) == ["train_loss", "val_dice"]

        with pytest.raises(ValueError, match="auxiliary classification head"):
            build_lightning_module("heatmap_3d", model=plain, loss=loss,
                                   optimizer=build_optimizer(plain.parameters(), name="adamw", lr=3e-4),
                                   cls_weight=0.003)

    def test_roc_auc_helper(self):
        from src.segmentation.lightning_module.heatmap_3d_module import _roc_auc

        assert _roc_auc([3, 2, 1], [1, 1, 0]) == 1.0
        assert _roc_auc([1, 2, 3], [1, 1, 0]) == 0.0
        assert _roc_auc([4, 3, 2, 1], [1, 0, 1, 0]) == 0.75
        assert _roc_auc([1, 1, 1, 1], [1, 1, 0, 0]) == 0.5   # a head collapsed to a constant scores 0.5
        assert _roc_auc([1, 2], [1, 1]) is None               # only one class present


class TestDynUNetAuxCls:
    """Auxiliary classification head, DynUNet variant.

    DynUNet's forward recurses through ``skip_layers``, so unlike SegResNet the
    bottleneck cannot be obtained by splitting ``encode``/``decode``. It is captured
    with a forward hook instead; these tests pin that the hook actually fires and
    that the seg path is unmodified.
    """

    KW = dict(in_channels=2, num_classes=1, kernel_size=[3, 3, 3, 3, 3],
              strides=[1, 2, 2, 2, 2], norm_name=["INSTANCE", {"affine": True}])

    @staticmethod
    def _batch(size: int = 32, b: int = 2):
        return {"image": torch.randn(b, 2, size, size, size)}

    def test_seg_path_identical_and_plain_ckpt_loads(self):
        """Plain monai_dynunet3d weights load as-is and the seg output is identical."""
        from src.segmentation.registries.models import build_model

        aux = build_model("monai_dynunet3d_auxcls", **self.KW)
        plain = build_model("monai_dynunet3d", **self.KW)
        missing, unexpected = aux.load_state_dict(plain.state_dict(), strict=False)
        assert not unexpected
        assert missing and all(m.startswith("cls_head") for m in missing)
        aux.eval(), plain.eval()
        batch = self._batch()
        with torch.no_grad():
            assert torch.equal(aux(batch)["masks"], plain(batch)["masks"])

    def test_cls_head_zero_init_starts_at_ln2(self):
        """Zero-initialised output => logits 0 => BCE starts at ln2, so cls_weight can be set arithmetically."""
        import math

        from src.segmentation.registries.models import build_model

        model = build_model("monai_dynunet3d_auxcls", **self.KW).eval()
        with torch.no_grad():
            logits = model(self._batch())["cls_logits"]
        assert logits.shape == (2, 1)
        assert torch.equal(logits, torch.zeros_like(logits))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0], torch.tensor([1.0, 0.0]))
        assert abs(float(bce) - math.log(2)) < 1e-5

    def test_encoder_gradient_is_two_stage(self):
        """Right after zero-init no gradient reaches the encoder; after one step it does (by design).

        It is 0 at step 0 because W_last=0 blocks backpropagation. This avoids an
        initial shock to the warm-started encoder, but it must not stay 0 (otherwise
        the head would never train the shared encoder), so both stages are pinned.
        """
        from src.segmentation.registries.models import build_model

        model = build_model("monai_dynunet3d_auxcls", **self.KW).train()
        batch = self._batch()

        def grads():
            model.zero_grad()
            model(batch)["cls_logits"].sum().backward()
            enc = sum(float(p.grad.abs().sum()) for p in model.net.bottleneck.parameters()
                      if p.grad is not None)
            return enc, float(model.cls_head[-1].weight.grad.abs().sum())

        enc0, last0 = grads()
        assert enc0 == 0.0            # output layer is zero, so nothing reaches the encoder
        assert last0 > 0.0            # the output layer itself still learns
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        enc1, _ = grads()
        assert enc1 > 0.0             # after the update it reaches the shared encoder

    def test_deep_supervision_still_splits_heads(self):
        """The masks/ds_masks/cls_logits contract holds together with deep_supervision."""
        from src.segmentation.registries.models import build_model

        model = build_model("monai_dynunet3d_auxcls", deep_supervision=True,
                            deep_supr_num=2, **self.KW).train()
        out = model(self._batch())
        assert out["masks"].shape == (2, 1, 32, 32, 32)
        assert out["ds_masks"].shape[:2] == (2, 2)
        assert out["cls_logits"].shape == (2, 1)
        model.eval()                  # aux heads disappear in eval (inference path unchanged)
        with torch.no_grad():
            assert "ds_masks" not in model(self._batch())


class TestBlobRelabelFromRefined:
    """Submission path: option to re-derive the location label from the Step4 refined-mask centroid.

    Default is False (centroid of the detected blob sphere, as before). Pins that
    behaviour only changes when enabled, and that the crop -> volume coordinate
    mapping is exact.
    """

    @staticmethod
    def _scene():
        """Volume with different vessel classes on each side; shifting the centroid right changes the label."""
        import numpy as np

        proba = np.zeros((60, 60, 60), np.float32)
        proba[28:33, 28:33, 28:33] = 0.9
        vessel = np.zeros((60, 60, 60), np.int64)
        vessel[:, :, :34], vessel[:, :, 34:] = 4, 8
        return proba, vessel, np.zeros((60, 60, 60), np.float32)

    class _MockFineSeg:
        """Returns a blob offset by +6 voxels from the blob centre (simulates a displaced aneurysm)."""

        size = 16

        def refine(self, image, vessel, center):
            import numpy as np

            from src.segmentation.inference.blob_mask import _crop_bounds

            half = self.size // 2
            dst, src = _crop_bounds(image.shape, center, half)
            m = np.zeros((self.size,) * 3, bool)
            m[half - 1:half + 2, half - 1:half + 2, half + 4:half + 7] = True
            return m, dst, src, 0.9        # 4th value is the fine-seg peak probability

    def test_crop_to_volume_mapping_is_exact(self):
        """The inverse of crop[src] = vol[dst], i.e. vol = crop - src.start + dst.start, is exact.

        Round-trips include crops that overflow the volume boundary. If this were
        off, relabelling would silently read vessels from the wrong location.
        """
        import numpy as np
        from scipy.ndimage import center_of_mass

        from src.segmentation.inference.blob_mask import _crop_bounds

        shape, half = (40, 40, 40), 8
        for pt in [(20, 20, 20), (3, 20, 37), (0, 39, 20)]:
            dst, src = _crop_bounds(shape, pt, half)
            marker = np.zeros(shape, bool)
            marker[pt] = True
            crop = np.zeros((2 * half,) * 3, bool)
            crop[src] = marker[dst]
            assert crop.any()
            cc = center_of_mass(crop)
            back = tuple(c - s.start + d.start for c, s, d in zip(cc, src, dst))
            assert np.allclose(back, pt)

    def test_default_is_unchanged(self):
        """Default (False) is unchanged and the relabel counters stay at zero."""
        from src.segmentation.inference.blob_mask import paint_blobs

        proba, vessel, image = self._scene()
        _, labels, stats = paint_blobs(proba, vessel, thr=0.5, min_vox=10, r_vox=4,
                                       image=image, fineseg=self._MockFineSeg())
        assert labels and stats["refined"] == 1
        assert stats["relabel_changed"] == stats["relabel_same"] == stats["relabel_none"] == 0

    def test_relabel_runs_on_refined_centroid(self):
        from src.segmentation.inference.blob_mask import paint_blobs

        proba, vessel, image = self._scene()
        _, labels, stats = paint_blobs(proba, vessel, thr=0.5, min_vox=10, r_vox=4,
                                       image=image, fineseg=self._MockFineSeg(),
                                       relabel_from_refined=True)
        assert labels
        # the labeller runs exactly once on the refined centroid (changed / same / no vessel)
        assert stats["relabel_changed"] + stats["relabel_same"] + stats["relabel_none"] == 1

    def test_relabel_never_drops_a_detection(self):
        """With no vessel evidence at the refined centroid, the detection is kept with its original label."""
        import numpy as np

        from src.segmentation.inference.blob_mask import paint_blobs

        proba, vessel, image = self._scene()
        vessel = np.zeros_like(vessel)
        # The blob centroid is at z=30, the refined centroid +5 at z=35. Restricting
        # vessels to z<31 gives evidence for the former (28-32 with r_vox=2) but not
        # for the latter (33-37).
        vessel[:, :, :31] = 4
        _, labels, stats = paint_blobs(proba, vessel, thr=0.5, min_vox=10, r_vox=2,
                                       image=image, fineseg=self._MockFineSeg(),
                                       relabel_from_refined=True)
        assert len(labels) == 1                       # the detection is kept
        assert stats["relabel_none"] == 1
        assert stats["relabel_changed"] == 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
