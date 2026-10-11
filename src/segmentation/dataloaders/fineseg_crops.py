"""Fine aneurysm-shape segmentation crops (detect -> crop -> fine seg, Task2 DSC).

Two parts:
  * ``crop_at`` / ``generate`` — offline crop generation. Around each crop center
    cut a fixed cube from (image, predicted vessel, GT location mask). Centers:
      - detector blob matched to a GT aneurysm (TP) -> target = true aneurysm shape
      - detector blob NOT matched (FP/reject)       -> target = (near-)empty
      - GT aneurysm the detector MISSED (FN)         -> target = true shape, cropped
        from the GT-mask center.
    Target is BINARY (aneurysm vs bg); the crop also stores the 50-location label
    for the optional classification head.
  * ``FineSegCropDataset`` (registered ``fineseg_crops``) — reads those crops for
    ``scripts/train_segmentation.py``. __getitem__ -> {"image": [1|2|38,S,S,S],
    "masks": [1,S,S,S] long, "cls_label", "sample_id"}.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.segmentation.registries.datasets import register_dataset

# 36-class vessel R/L swap pairs (same as Step2) for the aux channel on LR-mirror.
VESSEL_LR_PAIRS = [[2, 3], [4, 6], [5, 7], [8, 9], [11, 12], [13, 14], [17, 19],
                   [18, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                   [31, 32], [33, 34], [35, 36]]
# Location-id mapping, only used to mirror ``cls_label`` on LR flip. The label
# feeds the optional classification head; the binary segmentation target (the
# only thing the Step4 model is trained on) does not depend on it.
_LOCMAP = Path(os.environ.get("TOPANEU_DEPLOYMENT",
                             Path(__file__).resolve().parents[3] / "docker" / "mapping")) / "location_mapping.json"


# ----------------------------- crop generation -----------------------------
def crop_at(vol: np.ndarray, center, half: int, cval: float) -> np.ndarray:
    """Cut a ``(2*half)^3`` cube centered at ``center`` (padding OOB with ``cval``)."""
    c = [int(round(x)) for x in center]
    sl, pads = [], []
    for a in range(3):
        lo, hi = c[a] - half, c[a] + half
        pads.append((max(0, -lo), max(0, hi - vol.shape[a])))
        sl.append(slice(max(0, lo), min(vol.shape[a], hi)))
    sub = vol[tuple(sl)]
    if any(p != (0, 0) for p in pads):
        sub = np.pad(sub, pads, mode="constant", constant_values=cval)
    return sub


def generate(manifest_path, images_dir, vessel_dir, gt_dir, out_dir,
             size=64, tol_mm=5.0, spacing=0.4) -> dict:
    manifest = json.load(open(manifest_path))
    images_dir, vessel_dir, gt_dir, out_dir = map(Path, (images_dir, vessel_dir, gt_dir, out_dir))
    half = size // 2
    tolv = tol_mm / spacing
    for sub in ("images", "vessel", "target"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    entries = []
    n_tp = n_fp = n_fn = 0
    for cid, m in manifest.items():
        img = np.load(images_dir / f"{cid}.npy")
        ves = np.load(vessel_dir / f"{cid}.npy")
        gt_bin = (np.load(gt_dir / f"{cid}.npy") > 0).astype(np.uint8)
        blob_centers = [np.array(b["c"]) for b in m["blobs"]]

        # (center, kind, location_label): TP=matched GT loc, FP=0(none), FN=GT loc
        centers = []
        for b in m["blobs"]:
            centers.append((b["c"], "tp" if b["label"] > 0 else "fp", int(b["label"])))
            n_tp += b["label"] > 0
            n_fp += b["label"] == 0
        for g in m["gt"]:
            gc = np.array(g["c"])
            if not blob_centers or min(np.linalg.norm(gc - bc) for bc in blob_centers) > tolv:
                centers.append((g["c"], "fn", int(g["label"])))
                n_fn += 1

        for i, (ctr, kind, loc_label) in enumerate(centers):
            stem = f"{cid}__{i}"
            np.save(out_dir / "images" / f"{stem}.npy", crop_at(img, ctr, half, float(img.min())).astype(np.int16))
            np.save(out_dir / "vessel" / f"{stem}.npy", crop_at(ves, ctr, half, 0).astype(np.uint8))
            tgc = crop_at(gt_bin, ctr, half, 0).astype(np.uint8)
            np.save(out_dir / "target" / f"{stem}.npy", tgc)
            entries.append({"id": stem, "case": cid, "split": m["split"], "kind": kind,
                            "pos": int(tgc.sum() > 0), "label": loc_label})

    json.dump(entries, open(out_dir / "manifest.json", "w"))
    return {
        "n_crops": len(entries), "size": size, "tp": n_tp, "fp": n_fp, "fn": n_fn,
        "pos": sum(e["pos"] for e in entries),
        "train": sum(e["split"] == "train" for e in entries),
        "val": sum(e["split"] == "val" for e in entries),
    }


def _location_lr_lut(size=256):
    """LUT swapping each R-<x> location id with its L-<x> counterpart (central
    locations unchanged), so the classification label mirrors on LR flip."""
    labels = json.load(open(_LOCMAP))["labels"]
    base = {}
    for name, i in labels.items():
        if i == 0:
            continue
        side = "R" if name.startswith("R-") else "L" if name.startswith("L-") else None
        b = re.sub(r"^[RL]-", "", name)
        base.setdefault(b, {})[side] = i
    lut = np.arange(size, dtype=np.int64)
    for b, d in base.items():
        if "R" in d and "L" in d:
            lut[d["R"]], lut[d["L"]] = d["L"], d["R"]
    return lut


# ----------------------------- dataset -----------------------------
class FineSegCropDataset(Dataset):
    def __init__(
        self,
        crops_dir: str,
        num_classes: int = 2,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        aux_max: float = 36.0,
        aux_encoding: str = "scalar",   # "scalar" (label/aux_max, 1ch) | "onehot" (37ch)
        n_vessel: int = 37,
        lr_flip_p: float = 0.0,
        intensity_shift: float = 0.1,
        intensity_scale: float = 0.1,
        noise_std: float = 0.02,
        present_labels=None,   # list of present location ids -> compact classes (1..K); 0=none
        transforms=None,   # unused (kept for train-script injection compat)
        fold: int = 0,     # unused: split is baked into the fold-specific manifest
        is_train: bool = True,
        **kwargs,
    ):
        self.dir = Path(crops_dir)
        self.lo_p, self.hi_p, self.aux_max = lower_percentile, upper_percentile, float(aux_max)
        self.aux_encoding, self.n_vessel = aux_encoding, int(n_vessel)
        self.is_train = is_train
        self.lr_flip_p = lr_flip_p
        self.intensity_shift, self.intensity_scale, self.noise_std = intensity_shift, intensity_scale, noise_std
        self.aux_lut = np.arange(256, dtype=np.int64)
        for a, b in VESSEL_LR_PAIRS:
            self.aux_lut[a], self.aux_lut[b] = b, a
        self.loc_lut = _location_lr_lut()
        if isinstance(present_labels, str):
            present_labels = json.load(open(present_labels))
        self.loc2compact = ({int(l): i + 1 for i, l in enumerate(present_labels)}
                            if present_labels else None)
        split = "train" if is_train else "val"
        self.entries = [e for e in json.load(open(self.dir / "manifest.json")) if e["split"] == split]
        pos = sum(e["pos"] for e in self.entries)
        print(f"[FineSegCropDataset] {split}: {len(self.entries)} crops (pos={pos})")

    def __len__(self):
        return len(self.entries)

    def _norm(self, img):
        lo, hi = np.percentile(img, self.lo_p), np.percentile(img, self.hi_p)
        return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

    def __getitem__(self, idx):
        e = self.entries[idx]
        stem = e["id"]
        img = self._norm(np.load(self.dir / "images" / f"{stem}.npy").astype(np.float32))
        ves = np.load(self.dir / "vessel" / f"{stem}.npy").astype(np.int64)
        tgt = np.load(self.dir / "target" / f"{stem}.npy").astype(np.int64)
        cls_label = int(e.get("label", 0))

        if self.is_train and self.lr_flip_p > 0 and np.random.random() < self.lr_flip_p:
            img = np.ascontiguousarray(img[::-1])
            tgt = np.ascontiguousarray(tgt[::-1])                 # binary target: flip only
            ves = self.aux_lut[np.ascontiguousarray(ves[::-1])]   # vessel: flip + R/L swap
            cls_label = int(self.loc_lut[cls_label])              # location label: R<->L swap (orig space)
        if self.loc2compact is not None:                          # -> compact class (0=none)
            cls_label = self.loc2compact.get(cls_label, 0)
        if self.is_train:
            if self.intensity_scale or self.intensity_shift:
                img = img * (1 + np.random.uniform(-self.intensity_scale, self.intensity_scale)) \
                    + np.random.uniform(-self.intensity_shift, self.intensity_shift)
            if self.noise_std:
                img = img + np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = np.clip(img, 0, 1).astype(np.float32)

        img = np.ascontiguousarray(img)
        ves = np.ascontiguousarray(ves)
        if self.aux_encoding == "onehot":
            oh = np.eye(self.n_vessel, dtype=np.float32)[np.clip(ves, 0, self.n_vessel - 1)]
            image = np.concatenate([img[None], oh.transpose(3, 0, 1, 2)], 0)  # [1+n_vessel, ...]
        else:
            image = np.stack([img, ves.astype(np.float32) / self.aux_max], 0)  # [2, ...]
        return {
            "image": torch.from_numpy(image).float(),
            "masks": torch.from_numpy(np.ascontiguousarray(tgt))[None].long(),
            "cls_label": torch.tensor(cls_label, dtype=torch.long),
            "sample_id": stem,
        }


@register_dataset("fineseg_crops", aliases=["aneurysm_fineseg"])
def build_fineseg_crops(**kwargs) -> FineSegCropDataset:
    return FineSegCropDataset(**kwargs)
