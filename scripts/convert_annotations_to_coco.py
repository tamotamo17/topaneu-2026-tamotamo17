#!/usr/bin/env python
"""Convert your own bounding-box annotations into Roboflow-format COCO.

This is a **preprocessing recipe** for the detection pipeline. RF-DETR reads a
Roboflow-format COCO directory (train/valid/test, each with
``_annotations.coco.json`` + images). Rather than hand-writing that JSON, you
wrap your annotations in a tiny PyTorch-style dataset that yields the standard
detection sample dict, and let ``export_to_coco`` (in
``src/detection/dataloaders/datasets.py``) write the COCO files for you — it
handles the ``[x1,y1,x2,y2] -> [x,y,w,h]`` and ``0-indexed -> 1-indexed``
conversions and the Roboflow ``objects`` placeholder.

Input CSV format (one row per bounding box)::

    filepath,label,x1,y1,x2,y2[,split]
    images/img0.jpg,cat,10,10,90,120,train
    images/img0.jpg,dog,130,40,200,150,train
    images/img1.jpg,cat,5,5,80,80,valid
    ...

- ``filepath`` is relative to ``--images-root``. Rows with the same filepath are
  grouped into one image.
- ``label`` is a class NAME; the set of names becomes the class list
  (sorted, mapped to 0-indexed ids internally).
- ``x1,y1,x2,y2`` are absolute pixel corner coordinates.
- ``split`` is optional. If present it must be one of train/valid/test. If
  absent, images are split deterministically by ``--val-ratio`` / ``--test-ratio``.

Usage::

    PYTHONPATH=. uv run python scripts/convert_annotations_to_coco.py \
        --csv data/my_raw/annotations.csv \
        --images-root data/my_raw \
        --out data/my_coco

Then train (see docs/dataset_detection.md)::

    PYTHONPATH=. uv run python scripts/train_detection.py \
        --config configs/detection/step1_vessel_roi.yaml --fold 0 --gpu 0 \
        --dataset-dir data/my_coco --skip-dataset-build
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.detection.dataloaders.datasets import export_to_coco


class _AnnotationDataset(Dataset):
    """Minimal dataset yielding the export_to_coco sample contract.

    Each item is: {"image": PIL.Image, "boxes": [[x1,y1,x2,y2], ...],
                   "labels": [int 0-indexed, ...], "sample_id": str}
    """

    def __init__(self, images: list[dict], images_root: Path, label_to_id: dict[str, int]):
        self.images = images  # [{"filepath", "boxes", "labels"(names)}, ...]
        self.images_root = images_root
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        rec = self.images[idx]
        image = Image.open(self.images_root / rec["filepath"]).convert("RGB")
        return {
            "image": image,
            "boxes": rec["boxes"],  # [[x1,y1,x2,y2], ...]
            "labels": [self.label_to_id[name] for name in rec["labels"]],  # 0-indexed
            "sample_id": Path(rec["filepath"]).stem,
        }


def _load_csv(csv_path: Path):
    """Group CSV rows by image. Returns (per_split images, class_names)."""
    by_image: dict[str, dict] = {}
    image_split: dict[str, str] = {}
    labels_seen: set[str] = set()

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            fp = row["filepath"]
            box = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            label = row["label"]
            labels_seen.add(label)
            rec = by_image.setdefault(fp, {"filepath": fp, "boxes": [], "labels": []})
            rec["boxes"].append(box)
            rec["labels"].append(label)
            if row.get("split"):
                image_split[fp] = row["split"].strip()

    class_names = sorted(labels_seen)
    return by_image, image_split, class_names


def _assign_splits(filepaths, image_split, val_ratio, test_ratio, seed):
    """Use the CSV split column if present; otherwise deterministic ratio split."""
    if image_split:  # split column provided for at least some images
        splits = {"train": [], "valid": [], "test": []}
        for fp in filepaths:
            splits.setdefault(image_split.get(fp, "train"), []).append(fp)
        return splits

    order = list(filepaths)
    random.Random(seed).shuffle(order)
    n = len(order)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    return {
        "test": order[:n_test],
        "valid": order[n_test : n_test + n_val],
        "train": order[n_test + n_val :],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert bbox annotations CSV -> Roboflow COCO")
    ap.add_argument("--csv", required=True, help="Annotations CSV (filepath,label,x1,y1,x2,y2[,split])")
    ap.add_argument("--images-root", required=True, help="Root dir that CSV filepaths are relative to")
    ap.add_argument("--out", required=True, help="Output COCO root (train/valid/test created under it)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Val fraction when no split column")
    ap.add_argument("--test-ratio", type=float, default=0.1, help="Test fraction when no split column")
    ap.add_argument("--seed", type=int, default=42, help="Seed for the ratio split")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    out_root = Path(args.out)

    by_image, image_split, class_names = _load_csv(Path(args.csv))
    label_to_id = {name: i for i, name in enumerate(class_names)}  # 0-indexed
    splits = _assign_splits(list(by_image), image_split, args.val_ratio, args.test_ratio, args.seed)

    print(f"Classes ({len(class_names)}): {class_names}")
    for split in ("train", "valid", "test"):
        fps = splits.get(split, [])
        ds = _AnnotationDataset([by_image[fp] for fp in fps], images_root, label_to_id)
        if len(ds) == 0:
            print(f"[{split}] WARNING: 0 images (RF-DETR requires all 3 splits non-empty)")
            continue
        export_to_coco(ds, out_root / split, num_classes=len(class_names), class_names=class_names)
        print(f"[{split}] {len(ds)} images -> {out_root / split}")

    print(f"Done. COCO dataset at: {out_root}")
    print("num_classes for your YAML model section:", len(class_names))


if __name__ == "__main__":
    main()
