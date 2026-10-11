"""2.5D **multiclass** Segmentation Dataset for 3D volumes stored as ``.npy``.

Same 2.5D idea as :mod:`src.segmentation.dataloaders.slices_2_5d` (a window of
``num_slices`` adjacent slices stacked as channels, target = the center slice),
but the target is an **integer label map** with ``num_classes`` mutually
exclusive classes (background = 0). Used for:

- **Step2** multiclass vessel segmentation (36 vessel classes + background = 37).
- **Step3** aneurysm-location segmentation (50 location classes + background = 51).

Optionally a second volume (``aux_dir``, e.g. the multiclass vessel mask from
Step1/Step2) is windowed the same way and concatenated as extra input channels,
so a model can condition aneurysm localization on vessel anatomy. When ``aux_dir``
is set the model's ``in_channels`` must be ``num_slices * 2``; otherwise
``num_slices``.

On-disk layout (produced by ``scripts/preprocess_vessel_segmentation.py``)::

    images_dir/<case>.npy   # 3D volume, int16   [X, Y, Z]
    masks_dir/<case>.npy    # 3D label map, uint8 [X, Y, Z]  (0..num_classes-1)
    aux_dir/<case>.npy      # 3D label map, uint8 [X, Y, Z]  (optional)

The train/val split is resolved exactly like the binary dataset: from persisted
``<folds_dir>/fold<fold>.yaml`` (recommended) or a runtime dirs split. Aux paths
are derived from ``aux_dir`` by case id.

``__getitem__`` returns:
    {"image": Tensor[C, H, W],          # C = num_slices (+ num_slices if aux)
     "masks": Tensor[1, H, W] (long),   # integer label map of the center slice
     "sample_id": "<case>_z<center>"}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.segmentation.registries.datasets import register_dataset


class Slices25DMulticlassDataset(Dataset):
    """2.5D multiclass adjacent-slice dataset over 3D ``.npy`` volumes.

    Args:
        folds_dir: Directory of persisted ``fold<N>.yaml`` split files. When set,
            case list + image/mask paths for ``fold`` come from the YAML.
        images_dir / masks_dir: Fallback split (ignored when ``folds_dir`` set).
        aux_dir: Optional directory of auxiliary label volumes (e.g. vessel
            masks) keyed by case id; windowed and concatenated as input channels.
        aux_mode: ``"label"`` feeds the aux label id scaled to ``[0, 1]`` by
            ``aux_max``; ``"binary"`` feeds ``(aux > 0)``.
        aux_max: Divisor for ``aux_mode="label"`` (max vessel label = 36).
        num_slices: Odd number of adjacent slices stacked as input channels.
        slice_axis: Axis along which slices are taken (2 = Z for [X, Y, Z]).
        num_classes: Number of segmentation classes **including** background.
        transforms: Albumentations ``Compose`` applied as
            ``transforms(image=[H, W, C], mask=[H, W, 1+aux_k])``. The mask carries
            the integer label (channel 0) plus any aux channels so all share the
            same geometric augmentation (nearest, no intensity normalization).
        fold / is_train / num_folds / seed: split controls (see binary dataset).
        min_fg_pixels: Training only — drop center slices whose label map has
            fewer than this many non-background voxels (0 = keep every slice).
        slice_step: Stride between sampled center slices.
    """

    def __init__(
        self,
        folds_dir: str | None = None,
        images_dir: str | None = None,
        masks_dir: str | None = None,
        aux_dir: str | None = None,
        aux_mode: str = "label",
        aux_max: float = 36.0,
        num_slices: int = 5,
        slice_axis: int = 2,
        num_classes: int = 37,
        transforms=None,
        fold: int = 0,
        is_train: bool = True,
        num_folds: int = 5,
        seed: int = 42,
        min_fg_pixels: int = 0,
        slice_step: int = 1,
        neg_ratio: float = 0.0,
        lr_flip_p: float = 0.0,
        lr_flip_axis: int = 0,
        lr_swap_pairs: list | None = None,
        lr_aux_swap_pairs: list | None = None,
        return_skeleton: bool = False,
        binarize_fg: bool = False,
    ):
        if num_slices % 2 == 0:
            raise ValueError(f"num_slices must be odd, got {num_slices}")
        if num_classes < 2:
            raise ValueError(f"multiclass dataset needs num_classes >= 2, got {num_classes}")
        # Collapse all foreground labels to a single class (target = mask > 0).
        # Used for binary aneurysm "presence" segmentation while still feeding the
        # multiclass vessel mask as aux conditioning. num_classes should be 2.
        self.binarize_fg = binarize_fg
        if aux_mode not in ("label", "binary"):
            raise ValueError(f"aux_mode must be 'label' or 'binary', got {aux_mode!r}")

        # Left-right (mirror) flip with R/L label swap. The flip is along the 2D
        # slice's ``lr_flip_axis`` (0 = rows; for [X, Y] axial slices this is the
        # anatomical left-right axis). Because R-/L- vessels carry distinct
        # labels, a plain mirror would mislabel laterality — so ``lr_swap_pairs``
        # (and ``lr_aux_swap_pairs`` for the aux volume) remap the swapped sides.
        self.lr_flip_p = lr_flip_p
        self.lr_flip_axis = lr_flip_axis
        self.lr_lut = self._build_swap_lut(lr_swap_pairs)
        self.lr_aux_lut = self._build_swap_lut(lr_aux_swap_pairs if lr_aux_swap_pairs is not None else lr_swap_pairs)
        # Per-class GT skeleton (centerline) target for a Skeleton-Recall loss
        # term (RSNA-2025 1st place): rewards covering thin vessels' centerlines.
        self.return_skeleton = return_skeleton

        self.num_slices = num_slices
        self.half = num_slices // 2
        self.slice_axis = slice_axis
        self.num_classes = num_classes
        self.transforms = transforms
        self.is_train = is_train
        self.min_fg_pixels = min_fg_pixels
        self.slice_step = max(1, slice_step)
        self._seed = seed
        self.neg_ratio = neg_ratio
        self.aux_dir = Path(aux_dir) if aux_dir else None
        self.aux_mode = aux_mode
        self.aux_max = float(aux_max)

        # Resolve the split into a list of (case_id, image_path, mask_path).
        if folds_dir is not None:
            self.entries, source = self._entries_from_yaml(folds_dir, fold, is_train)
        elif images_dir is not None and masks_dir is not None:
            self.entries, source = self._entries_from_dirs(images_dir, masks_dir, fold, is_train, num_folds, seed)
        else:
            raise ValueError("Provide either `folds_dir` or both `images_dir` and `masks_dir`.")

        # Per-worker lazily-opened memmap handles.
        self._img_cache: dict[int, np.ndarray] = {}
        self._mask_cache: dict[int, np.ndarray] = {}
        self._aux_cache: dict[int, np.ndarray] = {}

        self.index = self._build_index()
        self._img_cache.clear()
        self._mask_cache.clear()
        self._aux_cache.clear()

        split = "train" if is_train else "valid"
        print(
            f"[Slices25DMulticlassDataset] fold={fold} {split} ({source}): {len(self.entries)} cases, "
            f"{len(self.index)} slices (K={num_slices}, axis={slice_axis}, C={num_classes}, "
            f"aux={'on' if self.aux_dir else 'off'}, min_fg={min_fg_pixels}, step={self.slice_step})"
        )

    @staticmethod
    def _skeletonize_labels(label2d: np.ndarray) -> np.ndarray:
        """Per-class 2D skeleton (centerline) of an integer label map.

        Returns an int map with the same class ids on skeleton pixels, 0 else.
        Each class is skeletonized within its bounding box for speed.
        """
        from skimage.morphology import skeletonize

        skel = np.zeros_like(label2d)
        for c in np.unique(label2d):
            if c == 0:
                continue
            ys, xs = np.where(label2d == c)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            s = skeletonize(label2d[y0:y1, x0:x1] == c)
            skel[y0:y1, x0:x1][s] = c
        return skel

    @staticmethod
    def _build_swap_lut(pairs, size: int = 256):
        """Lookup table that swaps each (a, b) label pair; identity elsewhere."""
        if not pairs:
            return None
        lut = np.arange(size, dtype=np.int64)
        for a, b in pairs:
            lut[int(a)], lut[int(b)] = int(b), int(a)
        return lut

    # -- split resolution -------------------------------------------------------

    @staticmethod
    def _entries_from_yaml(folds_dir, fold, is_train):
        fold_path = Path(folds_dir) / f"fold{fold}.yaml"
        if not fold_path.exists():
            raise FileNotFoundError(f"Fold split file not found: {fold_path}")
        doc = OmegaConf.load(fold_path)
        rows = doc.train if is_train else doc.val
        entries = [(r["id"], Path(r["image"]), Path(r["mask"])) for r in rows]
        if not entries:
            raise RuntimeError(f"No {'train' if is_train else 'val'} cases in {fold_path}")
        return entries, f"yaml:{fold_path.name}"

    def _entries_from_dirs(self, images_dir, masks_dir, fold, is_train, num_folds, seed):
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        cases = sorted(p.stem for p in images_dir.glob("*.npy") if (masks_dir / f"{p.stem}.npy").exists())
        if not cases:
            raise RuntimeError(f"No paired .npy volumes found in {images_dir} / {masks_dir}")
        order = np.arange(len(cases))
        np.random.default_rng(seed).shuffle(order)
        fold_of = np.empty(len(cases), dtype=int)
        for rank, case_idx in enumerate(order):
            fold_of[case_idx] = rank % num_folds
        selected = [cases[i] for i in range(len(cases)) if (fold_of[i] != fold) == is_train]
        entries = [(c, images_dir / f"{c}.npy", masks_dir / f"{c}.npy") for c in selected]
        return entries, "dirs"

    # -- lazy volume access -----------------------------------------------------

    def _img(self, ci: int) -> np.ndarray:
        arr = self._img_cache.get(ci)
        if arr is None:
            arr = np.load(self.entries[ci][1], mmap_mode="r")
            self._img_cache[ci] = arr
        return arr

    def _mask(self, ci: int) -> np.ndarray:
        arr = self._mask_cache.get(ci)
        if arr is None:
            arr = np.load(self.entries[ci][2], mmap_mode="r")
            self._mask_cache[ci] = arr
        return arr

    def _aux(self, ci: int) -> np.ndarray | None:
        if self.aux_dir is None:
            return None
        arr = self._aux_cache.get(ci)
        if arr is None:
            aux_path = self.aux_dir / f"{self.entries[ci][0]}.npy"
            if not aux_path.exists():
                raise FileNotFoundError(f"aux volume not found: {aux_path}")
            arr = np.load(aux_path, mmap_mode="r")
            self._aux_cache[ci] = arr
        return arr

    def _take(self, arr: np.ndarray, idx: int) -> np.ndarray:
        idx = min(max(idx, 0), arr.shape[self.slice_axis] - 1)
        return np.take(arr, idx, axis=self.slice_axis)

    # -- index construction -----------------------------------------------------

    def _build_index(self) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        rng = np.random.default_rng(self._seed)
        for ci in range(len(self.entries)):
            mask = self._mask(ci)
            n = mask.shape[self.slice_axis]

            fg = None
            if self.is_train and (self.min_fg_pixels > 0 or self.neg_ratio > 0):
                reduce_axes = tuple(a for a in range(mask.ndim) if a != self.slice_axis)
                fg = (np.asarray(mask) > 0).sum(axis=reduce_axes)

            steps = list(range(0, n, self.slice_step))
            if fg is not None and self.neg_ratio > 0 and self.min_fg_pixels > 0:
                # Keep all foreground slices + a matched number of sampled empty
                # slices (teaches the model to suppress non-lesion regions, cutting
                # false positives for sparse targets like aneurysms).
                pos = [s for s in steps if fg[s] >= self.min_fg_pixels]
                neg = [s for s in steps if fg[s] == 0]
                k = min(len(neg), int(round(len(pos) * self.neg_ratio)))
                keep = pos + (list(rng.choice(neg, size=k, replace=False)) if k else [])
                index.extend((ci, int(s)) for s in sorted(keep))
            else:
                for s in steps:
                    if fg is not None and fg[s] < self.min_fg_pixels:
                        continue
                    index.append((ci, s))
        return index

    # -- Dataset protocol -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ci, center = self.index[idx]
        img = self._img(ci)
        mask = self._mask(ci)

        # [H, W, num_slices] window of adjacent image slices as channels.
        offsets = range(-self.half, self.half + 1)
        image = np.stack([self._take(img, center + off) for off in offsets], axis=-1).astype(np.float32)

        # Center-slice integer label map -> [H, W].
        label2d = np.asarray(self._take(mask, center)).astype(np.int64)
        if self.binarize_fg:
            label2d = (label2d > 0).astype(np.int64)

        aux_vol = self._aux(ci)
        aux_raw = None
        if aux_vol is not None:
            aux_raw = np.stack([self._take(aux_vol, center + off) for off in offsets], axis=-1).astype(np.int64)

        # Left-right mirror (train-time). The target label is swapped R<->L only
        # if lr_swap_pairs was given (multiclass vessel); a binary aneurysm target
        # needs no swap. The vessel aux is always swapped via lr_aux_swap_pairs.
        if self.is_train and self.lr_flip_p > 0 and np.random.random() < self.lr_flip_p:
            image = np.flip(image, axis=self.lr_flip_axis)
            label2d = np.flip(label2d, axis=self.lr_flip_axis)
            if self.lr_lut is not None:
                label2d = self.lr_lut[label2d]
            if aux_raw is not None:
                aux_raw = np.flip(aux_raw, axis=self.lr_flip_axis)
                if self.lr_aux_lut is not None:
                    aux_raw = self.lr_aux_lut[aux_raw]
        image = np.ascontiguousarray(image)

        # -> [H, W, 1] (float for transform). Channel order in mask_stack:
        # [label, (skeleton), (aux...)] so all share the geometric augmentation.
        label = np.ascontiguousarray(label2d).astype(np.float32)[..., None]
        parts = [label]
        if self.return_skeleton:
            parts.append(self._skeletonize_labels(np.ascontiguousarray(label2d)).astype(np.float32)[..., None])
        if aux_raw is not None:
            aux = np.ascontiguousarray(aux_raw).astype(np.float32)
            aux = (aux > 0).astype(np.float32) if self.aux_mode == "binary" else aux / self.aux_max
            parts.append(aux)
        mask_stack = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)

        if self.transforms is not None:
            out = self.transforms(image=image, mask=mask_stack)
            image = out["image"]
            mask_stack = out["mask"]

        # Image -> [C, H, W]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        else:  # ToTensorV2 already gives [C, H, W]
            image = image.float()

        # mask_stack -> [1 + aux_k, H, W]; ToTensorV2 keeps mask as [H, W, C].
        if isinstance(mask_stack, np.ndarray):
            mask_stack = torch.from_numpy(mask_stack.transpose(2, 0, 1).copy())
        elif mask_stack.ndim == 3:
            mask_stack = mask_stack.permute(2, 0, 1)
        else:  # [H, W] single channel
            mask_stack = mask_stack.unsqueeze(0)

        # Split channels back out in the order they were stacked.
        label_t = mask_stack[:1].long()  # [1, H, W] integer labels
        ch = 1
        out = {"image": image, "masks": label_t, "sample_id": f"{self.entries[ci][0]}_z{center:04d}"}
        if self.return_skeleton:
            out["skeleton"] = mask_stack[ch:ch + 1].long()  # [1, H, W] centerline labels
            ch += 1
        if aux_vol is not None:
            aux_t = mask_stack[ch:].float()  # [num_slices, H, W]
            out["image"] = torch.cat([image, aux_t], dim=0)
        return out


@register_dataset("slices_2_5d_mc", aliases=["seg_2_5d_mc", "slice25d_mc"])
def build_slices_2_5d_multiclass_dataset(
    folds_dir: str | None = None,
    images_dir: str | None = None,
    masks_dir: str | None = None,
    aux_dir: str | None = None,
    aux_mode: str = "label",
    aux_max: float = 36.0,
    num_slices: int = 5,
    slice_axis: int = 2,
    num_classes: int = 37,
    transforms=None,
    fold: int = 0,
    is_train: bool = True,
    num_folds: int = 5,
    seed: int = 42,
    min_fg_pixels: int = 0,
    slice_step: int = 1,
    neg_ratio: float = 0.0,
    lr_flip_p: float = 0.0,
    lr_flip_axis: int = 0,
    lr_swap_pairs: list | None = None,
    lr_aux_swap_pairs: list | None = None,
    return_skeleton: bool = False,
    binarize_fg: bool = False,
    **kwargs,
) -> Slices25DMulticlassDataset:
    """Build a :class:`Slices25DMulticlassDataset` (see its docstring)."""
    return Slices25DMulticlassDataset(
        neg_ratio=neg_ratio,
        lr_flip_p=lr_flip_p,
        lr_flip_axis=lr_flip_axis,
        lr_swap_pairs=lr_swap_pairs,
        lr_aux_swap_pairs=lr_aux_swap_pairs,
        return_skeleton=return_skeleton,
        binarize_fg=binarize_fg,
        folds_dir=folds_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
        aux_dir=aux_dir,
        aux_mode=aux_mode,
        aux_max=aux_max,
        num_slices=num_slices,
        slice_axis=slice_axis,
        num_classes=num_classes,
        transforms=transforms,
        fold=fold,
        is_train=is_train,
        num_folds=num_folds,
        seed=seed,
        min_fg_pixels=min_fg_pixels,
        slice_step=slice_step,
    )
