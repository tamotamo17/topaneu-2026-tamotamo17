"""3D patch dataset for volumetric multiclass vessel segmentation.

Train: samples ``patch_size`` cubes from the preprocessed crop volumes with a
foreground bias, applies a left-right mirror with R/L label swap and light
intensity augmentation, and (optionally) computes the per-class **3D** skeleton
for a Skeleton-Recall loss — a true centerline, unlike the degenerate per-slice
2.5D skeleton.

Val: returns the full volume (``is_train=False``) for sliding-window inference in
the Lightning module.

Split resolution reuses the persisted ``fold<N>.yaml`` (same as the 2.5D dataset).

__getitem__ returns:
    train: {"image": [1,D,H,W], "masks": [1,D,H,W] long,
            "skeleton": [1,D,H,W] long (optional), "sample_id"}
    val:   {"image": [1,X,Y,Z], "masks": [1,X,Y,Z] long, "sample_id": "<case>"}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.segmentation.registries.datasets import register_dataset


class Patches3DDataset(Dataset):
    def __init__(
        self,
        folds_dir: str | None = None,
        images_dir: str | None = None,
        masks_dir: str | None = None,
        num_classes: int = 37,
        patch_size: int = 96,
        samples_per_volume: int = 8,
        fg_prob: float = 0.8,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        transforms=None,       # unused (kept for train-script injection compat)
        fold: int = 0,
        is_train: bool = True,
        num_folds: int = 5,
        seed: int = 42,
        lr_flip_p: float = 0.0,
        lr_swap_pairs: list | None = None,
        lr_aux_swap_pairs: list | None = None,
        aux_dir: str | None = None,
        aux_mode: str = "label",
        aux_max: float = 36.0,
        aux_encoding: str = "scalar",   # "scalar" (label/aux_max, 1ch) | "onehot" (n_vessel ch)
        n_vessel: int = 37,
        soft_target: bool = False,      # True => mask is a continuous [0,1] heatmap (float), not class labels
        # --- foreground patch sampling (defaults reproduce the original behaviour) ---
        fg_center_thr: float = 0.0,     # candidate voxels are mask > thr. For a Gaussian heatmap the
                                        # default 0 spans the whole 3-sigma tail, so the "foreground"
                                        # voxel can be far from the lesion; 0.9 keeps it near the peak.
        fg_jitter: float = 0.0,         # 0 = patch centred on that voxel; 1 = voxel placed uniformly
                                        # anywhere inside the patch (translation diversity)
        fg_margin_vox: int = 0,         # keep the voxel at least this many voxels from the patch border
        # --- geometric augmentation (defaults = off, i.e. plain axis-aligned crop) ---
        rot_p: float = 0.0,             # probability of applying a random 3D rotation
        rot_deg: float = 30.0,          # per-axis rotation range, +-deg
        scale_p: float = 0.0,           # probability of applying a random isotropic scale
        scale_range: tuple = (0.7, 1.4),  # log-uniform; >1 zooms out (more context)
        intensity_shift: float = 0.1,
        intensity_scale: float = 0.1,
        noise_std: float = 0.02,
        return_skeleton: bool = False,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.ps = patch_size
        self.samples_per_volume = samples_per_volume
        self.fg_prob = fg_prob
        self.fg_center_thr = float(fg_center_thr)
        self.fg_jitter = float(fg_jitter)
        self.fg_margin_vox = int(fg_margin_vox)
        self._vmin_cache: dict[int, float] = {}
        self.rot_p, self.rot_deg = float(rot_p), float(rot_deg)
        self.scale_p, self.scale_range = float(scale_p), tuple(scale_range)
        self.lo_p, self.hi_p = lower_percentile, upper_percentile
        self.is_train = is_train
        self.lr_flip_p = lr_flip_p
        self.lr_lut = self._build_swap_lut(lr_swap_pairs)
        self.aux_lut = self._build_swap_lut(lr_aux_swap_pairs)
        self.aux_dir = Path(aux_dir) if aux_dir else None
        self.aux_mode, self.aux_max = aux_mode, float(aux_max)
        self.aux_encoding, self.n_vessel = aux_encoding, int(n_vessel)
        self.soft_target = soft_target
        self._mdt = np.float32 if soft_target else np.int64   # mask array dtype
        self.intensity_shift, self.intensity_scale, self.noise_std = intensity_shift, intensity_scale, noise_std
        self.return_skeleton = return_skeleton

        if folds_dir is not None:
            fold_path = Path(folds_dir) / f"fold{fold}.yaml"
            doc = OmegaConf.load(fold_path)
            rows = doc.train if is_train else doc.val
            self.entries = [(r["id"], Path(r["image"]), Path(r["mask"])) for r in rows]
        elif images_dir is not None and masks_dir is not None:
            images_dir, masks_dir = Path(images_dir), Path(masks_dir)
            cases = sorted(p.stem for p in images_dir.glob("*.npy") if (masks_dir / f"{p.stem}.npy").exists())
            order = np.arange(len(cases)); np.random.default_rng(seed).shuffle(order)
            fold_of = {cases[order[r]]: r % num_folds for r in range(len(cases))}
            sel = [c for c in cases if (fold_of[c] != fold) == is_train]
            self.entries = [(c, images_dir / f"{c}.npy", masks_dir / f"{c}.npy") for c in sel]
        else:
            raise ValueError("Provide folds_dir or images_dir+masks_dir")

        self._img: dict[int, np.ndarray] = {}
        self._msk: dict[int, np.ndarray] = {}
        self._aux: dict[int, np.ndarray] = {}
        self._norm: dict[int, tuple] = {}
        print(f"[Patches3DDataset] fold={fold} {'train' if is_train else 'valid'}: {len(self.entries)} cases "
              f"(patch={patch_size}, spv={samples_per_volume}, fg_prob={fg_prob}, "
              f"aux={'on' if self.aux_dir else 'off'}, skel={return_skeleton})")

    @staticmethod
    def _build_swap_lut(pairs, size=256):
        if not pairs:
            return None
        lut = np.arange(size, dtype=np.int64)
        for a, b in pairs:
            lut[int(a)], lut[int(b)] = int(b), int(a)
        return lut

    def _load(self, ci):
        if ci not in self._img:
            self._img[ci] = np.load(self.entries[ci][1], mmap_mode="r")
            self._msk[ci] = np.load(self.entries[ci][2], mmap_mode="r")
            img = np.asarray(self._img[ci], dtype=np.float32)
            self._norm[ci] = (float(np.percentile(img, self.lo_p)), float(np.percentile(img, self.hi_p)))
            if self.aux_dir is not None:
                self._aux[ci] = np.load(self.aux_dir / f"{self.entries[ci][0]}.npy", mmap_mode="r")
        return self._img[ci], self._msk[ci], self._aux.get(ci), self._norm[ci]

    def _encode_aux(self, aux_labels):
        """Encode a vessel-LABEL volume to channel-first aux: scalar (1ch) or
        one-hot (n_vessel ch). Keeps categorical identity when one-hot."""
        a = np.asarray(aux_labels)
        if self.aux_encoding == "onehot":
            oh = np.eye(self.n_vessel, dtype=np.float32)[np.clip(a.astype(np.int64), 0, self.n_vessel - 1)]
            return oh.transpose(3, 0, 1, 2)                       # [C, D, H, W]
        if self.aux_mode == "binary":
            return (a > 0).astype(np.float32)[None]               # [1, D, H, W]
        return (a.astype(np.float32) / self.aux_max)[None]        # [1, D, H, W]

    @staticmethod
    def _apply_norm(img, lo, hi):
        # Per-VOLUME percentile (consistent between patch-train and full-volume
        # val — normalizing per-patch instead collapses sliding-window val).
        return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

    def __len__(self):
        return len(self.entries) * self.samples_per_volume if self.is_train else len(self.entries)

    # -- validation: full volume ------------------------------------------------
    def _get_val(self, ci):
        img_v, msk_v, aux_v, (lo, hi) = self._load(ci)
        img = self._apply_norm(np.asarray(img_v, dtype=np.float32), lo, hi)
        msk = np.asarray(msk_v, dtype=self._mdt)
        if aux_v is None:
            image = img[None]
        else:
            image = np.concatenate([img[None], self._encode_aux(aux_v)], 0)
        return {
            "image": torch.from_numpy(image).float(),
            "masks": torch.from_numpy(msk[None]).float() if self.soft_target else torch.from_numpy(msk[None]).long(),
            "sample_id": self.entries[ci][0],
        }

    # -- training: foreground-biased patch --------------------------------------

    def _vmin(self, vol):
        """Volume minimum, cached: scanning a 300^3 volume per patch costs ~2ms."""
        key = id(vol)
        if key not in self._vmin_cache:
            self._vmin_cache[key] = float(np.asarray(vol).min())
        return self._vmin_cache[key]

    def _rand_affine(self):
        """Random rotation x isotropic scale as a 3x3 matrix mapping OUT -> IN coords."""
        m = np.eye(3, dtype=np.float64)
        if self.rot_p > 0 and np.random.random() < self.rot_p:
            for axis in range(3):
                a = np.deg2rad(np.random.uniform(-self.rot_deg, self.rot_deg))
                c, s_ = np.cos(a), np.sin(a)
                r = np.eye(3)
                u, v = [k for k in range(3) if k != axis]
                r[u, u], r[u, v], r[v, u], r[v, v] = c, -s_, s_, c
                m = m @ r
        if self.scale_p > 0 and np.random.random() < self.scale_p:
            lo, hi = self.scale_range
            m = m * float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
        return m

    def _affine_patch(self, img_v, msk_v, aux_v, anchor_in, anchor_out, m):
        """Resample a patch so volume point ``anchor_in`` lands on patch point
        ``anchor_out``, under the affine ``m`` (OUT -> IN coordinates).

        Sampling straight from the volume (instead of cropping then rotating) means
        rotated patches have no empty corners: the interpolator reads further out
        in the volume. Only the source bounding box actually touched is
        materialised — feeding the whole 300^3 volume costs ~30ms per patch in
        dtype conversion alone and hurts cache locality in the interpolator.
        """
        ps = self.ps
        offset = np.asarray(anchor_in, dtype=np.float64) - m @ np.asarray(anchor_out, dtype=np.float64)
        out = (ps, ps, ps)
        # bounding box of the 8 transformed patch corners, clipped to the volume
        corners = np.array([[x, y, z] for x in (0, ps) for y in (0, ps) for z in (0, ps)], dtype=np.float64)
        src = corners @ m.T + offset
        lo = np.maximum(np.floor(src.min(0)).astype(int) - 1, 0)
        hi = np.minimum(np.ceil(src.max(0)).astype(int) + 2, np.asarray(img_v.shape))
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        off = offset - lo
        img = affine_transform(np.asarray(img_v[sl], dtype=np.float32), m, offset=off, output_shape=out,
                               order=1, mode="constant", cval=self._vmin(img_v))
        msk = affine_transform(np.asarray(msk_v[sl], dtype=self._mdt), m, offset=off, output_shape=out,
                               order=1 if self.soft_target else 0, mode="constant", cval=0)
        aux = None if aux_v is None else affine_transform(
            np.asarray(aux_v[sl]), m, offset=off, output_shape=out, order=0, mode="constant", cval=0)
        return img.astype(np.float32), msk.astype(self._mdt), (None if aux is None else aux.astype(np.float32))

    def _sample_patch_affine(self, img, msk, aux):
        """Foreground/random anchor selection identical to :meth:`_sample_patch`,
        but the patch is resampled with a random rotation/scale instead of sliced."""
        ps = self.ps
        half = ps // 2
        fg_vox = None
        if np.random.random() < self.fg_prob:
            nz = np.argwhere(np.asarray(msk) > self.fg_center_thr)
            if len(nz) == 0:
                nz = np.argwhere(np.asarray(msk) > 0)
            if len(nz):
                fg_vox = nz[np.random.randint(len(nz))]
        if fg_vox is None:
            anchor_in = [np.random.uniform(half, max(half + 1, msk.shape[a] - half)) for a in range(3)]
            anchor_out = [half] * 3
        else:
            max_off = max(0, half - self.fg_margin_vox)
            anchor_in = [float(fg_vox[a]) for a in range(3)]
            anchor_out = [half + (int(round(np.random.uniform(-1.0, 1.0) * self.fg_jitter * max_off))
                                  if self.fg_jitter > 0 else 0) for _ in range(3)]
        m = self._rand_affine()
        if np.allclose(m, np.eye(3)):
            # nothing to warp (both draws missed) -> plain slicing is ~200x cheaper
            start = [int(np.clip(round(anchor_in[a] - anchor_out[a]), 0, max(0, msk.shape[a] - ps)))
                     for a in range(3)]
            sl = tuple(slice(st, st + ps) for st in start)
            return (np.asarray(img[sl], dtype=np.float32), np.asarray(msk[sl], dtype=self._mdt),
                    None if aux is None else np.asarray(aux[sl], dtype=np.float32))
        return self._affine_patch(img, msk, aux, anchor_in, anchor_out, m)

    def _sample_patch(self, img, msk, aux):
        """Cut one patch, optionally guaranteed to contain a foreground voxel.

        The foreground voxel is drawn from ``mask > fg_center_thr``. With a
        Gaussian heatmap target the default threshold of 0 spans the entire
        3-sigma tail, so raising sigma silently widens this region: measured on
        the 0.4mm crops, a sigma=5mm target put the lesion centre a median 14.7mm
        (max 28.1mm) from the patch centre, i.e. partly OUTSIDE a 128^3 patch.
        Setting ``fg_center_thr`` near the peak keeps the drawn voxel on the
        lesion regardless of sigma.

        ``fg_jitter`` then decides where inside the patch that voxel lands: 0
        pins it to the patch centre (the original behaviour, which gives the
        network no translation diversity), 1 spreads it uniformly across the
        patch minus ``fg_margin_vox``. The voxel is always inside the patch.
        """
        ps = self.ps
        shape = img.shape
        start = []
        fg = np.random.random() < self.fg_prob
        fg_vox = None
        if fg:
            nz = np.argwhere(np.asarray(msk) > self.fg_center_thr)
            if len(nz) == 0:                      # threshold too strict for this volume -> fall back
                nz = np.argwhere(np.asarray(msk) > 0)
            if len(nz):
                fg_vox = nz[np.random.randint(len(nz))]
        for a in range(3):
            if shape[a] <= ps:
                s = 0
            elif fg_vox is not None:
                c = int(fg_vox[a])
                half = ps // 2
                # how far the voxel may sit from the patch centre and stay in-bounds
                max_off = max(0, half - self.fg_margin_vox)
                off = (int(round(np.random.uniform(-1.0, 1.0) * self.fg_jitter * max_off))
                       if self.fg_jitter > 0 else 0)
                s = int(np.clip(c - half + off, 0, shape[a] - ps))
            else:
                s = np.random.randint(0, shape[a] - ps + 1)
            start.append(s)
        sl = tuple(slice(s, s + ps) for s in start)
        aux_p = None if aux is None else np.asarray(aux[sl], dtype=np.float32)
        return np.asarray(img[sl], dtype=np.float32), np.asarray(msk[sl], dtype=self._mdt), aux_p

    @staticmethod
    def _pad_to(arr, ps, cval):
        pads = [(0, max(0, ps - arr.shape[a])) for a in range(3)]
        return np.pad(arr, pads, mode="constant", constant_values=cval) if any(p[1] for p in pads) else arr

    def _skeleton3d(self, patch_msk):
        from skimage.morphology import skeletonize
        skel = np.zeros_like(patch_msk)
        for c in np.unique(patch_msk):
            if c == 0:
                continue
            m = patch_msk == c
            idx = np.argwhere(m)
            mn, mx = idx.min(0), idx.max(0) + 1
            sub = m[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
            s = skeletonize(sub)
            skel[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]][s] = c
        return skel

    def __getitem__(self, idx):
        ci = idx % len(self.entries) if not self.is_train else np.random.randint(len(self.entries))
        img_v, msk_v, aux_v, (lo, hi) = self._load(ci)
        if not self.is_train:
            return self._get_val(ci)

        if self.rot_p > 0 or self.scale_p > 0:
            img, msk, aux = self._sample_patch_affine(img_v, msk_v, aux_v)
        else:
            img, msk, aux = self._sample_patch(img_v, msk_v, aux_v)
        img = self._pad_to(img, self.ps, float(img.min())); msk = self._pad_to(msk, self.ps, 0)
        img = self._apply_norm(img, lo, hi)  # per-volume percentile (matches val)
        if aux is not None:
            aux = self._pad_to(aux, self.ps, 0).astype(np.int64)   # keep as vessel LABELS

        # Left-right mirror (axis 0 = X) with R/L label swap (mask + aux vessel).
        if self.lr_flip_p > 0 and np.random.random() < self.lr_flip_p:
            img = np.ascontiguousarray(img[::-1])
            if self.lr_lut is not None:
                msk = self.lr_lut[np.ascontiguousarray(msk[::-1])]
            else:
                msk = np.ascontiguousarray(msk[::-1])
            if aux is not None:
                aux = np.ascontiguousarray(aux[::-1])
                if self.aux_lut is not None and self.aux_mode == "label":
                    aux = self.aux_lut[aux]                         # swap R/L on labels directly

        # Light intensity augmentation (on [0,1]).
        if self.intensity_scale or self.intensity_shift:
            img = img * (1 + np.random.uniform(-self.intensity_scale, self.intensity_scale)) \
                + np.random.uniform(-self.intensity_shift, self.intensity_shift)
        if self.noise_std:
            img = img + np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1).astype(np.float32)

        img = np.ascontiguousarray(img)
        if aux is None:
            image = img[None]
        else:
            image = np.concatenate([img[None], self._encode_aux(np.ascontiguousarray(aux))], 0)
        out = {
            "image": torch.from_numpy(image).float(),
            "masks": (torch.from_numpy(np.ascontiguousarray(msk)[None]).float() if self.soft_target else torch.from_numpy(np.ascontiguousarray(msk)[None]).long()),
            "sample_id": f"{self.entries[ci][0]}_p{idx}",
        }
        if self.return_skeleton:
            out["skeleton"] = torch.from_numpy(self._skeleton3d(np.ascontiguousarray(msk))[None]).long()
        return out


@register_dataset("patches_3d", aliases=["patch3d", "seg_3d"])
def build_patches_3d(**kwargs) -> Patches3DDataset:
    return Patches3DDataset(**kwargs)
