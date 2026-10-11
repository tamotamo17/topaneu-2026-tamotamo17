"""Detection Datasets.

Provides dataset implementations for object detection:
- RandomDetectionDataset: Dummy dataset for testing
- COCODetectionDataset: COCO format dataset parser
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.detection.registries.datasets import register_dataset


class RandomDetectionDataset(Dataset):
    """Random detection dataset for testing and debugging.

    Generates random images with random bounding boxes and labels.
    Useful for quick testing without real data.

    Args:
        num_samples: Number of samples in the dataset
        num_classes: Number of detection classes
        image_size: Image size (square)
        max_boxes_per_image: Maximum number of boxes per image
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 6,
        image_size: int = 640,
        max_boxes_per_image: int = 10,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.max_boxes_per_image = max_boxes_per_image
        self.seed = seed
        # Reproducibility is handled per-sample in __getitem__ via a local
        # np.random.RandomState(self.seed + idx). We deliberately do NOT call the
        # global random.seed()/np.random.seed() here, as that would clobber the
        # process-wide RNG state set by src.common.seed.set_seed().

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a random detection sample.

        Returns:
            dict with keys:
                - image: PIL Image
                - boxes: List of [x1, y1, x2, y2] boxes
                - labels: List of class IDs (0-indexed)
                - sample_id: Sample identifier
        """
        # Set per-sample seed for reproducibility
        rng = np.random.RandomState(self.seed + idx)

        image_array = rng.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)

        # Generate random number of boxes (1 to max_boxes_per_image)
        num_boxes = rng.randint(1, self.max_boxes_per_image + 1)

        boxes = []
        labels = []

        for _ in range(num_boxes):
            # Random box coordinates (x1, y1, x2, y2)
            x1 = rng.randint(0, self.image_size - 20)
            y1 = rng.randint(0, self.image_size - 20)
            x2 = rng.randint(x1 + 10, min(x1 + 100, self.image_size))
            y2 = rng.randint(y1 + 10, min(y1 + 100, self.image_size))

            boxes.append([x1, y1, x2, y2])

            # Random class label (0-indexed)
            label = rng.randint(0, self.num_classes)
            labels.append(label)

        return {
            "image": image,
            "boxes": boxes,  # List of [x1, y1, x2, y2]
            "labels": labels,  # List of class IDs (0-indexed)
            "sample_id": f"random_{idx:05d}",
        }


class COCODetectionDataset(Dataset):
    """COCO format detection dataset.

    Expects the canonical Roboflow-format layout, with images placed directly
    under the split directory (this matches ``export_to_coco`` /
    the projection data generator and what RF-DETR's ``dataset_file="roboflow"``
    reads)::

        coco_root/
        ├── split/
        │   ├── _annotations.coco.json
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...

    For backward compatibility, an ``images/`` subdirectory is also searched if
    a ``file_name`` is not found directly under the split directory.

    Args:
        coco_root: Root directory containing COCO dataset
        split: Dataset split ('train', 'valid', or 'test')
        transform: Optional image transforms
    """

    def __init__(
        self,
        coco_root: str | Path,
        split: str = "train",
        transform: Any = None,
    ):
        self.coco_root = Path(coco_root)
        self.split = split
        self.transform = transform

        self.split_dir = self.coco_root / split
        # Canonical layout puts images directly under the split dir; an
        # ``images/`` subdir is kept as a backward-compatible fallback.
        self.images_dir = self.split_dir / "images"
        self.annotations_file = self.split_dir / "_annotations.coco.json"

        if not self.annotations_file.exists():
            raise FileNotFoundError(
                f"COCO annotations not found: {self.annotations_file}\n"
                f"Expected structure (Roboflow format, images directly under the split):\n"
                f"  {coco_root}/\n"
                f"    {split}/\n"
                f"      _annotations.coco.json\n"
                f"      <image>.jpg ...\n"
            )

        with open(self.annotations_file) as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}
        self.image_ids = sorted(self.images.keys())

        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _resolve_image_path(self, file_name: str) -> Path:
        """Resolve a COCO ``file_name`` to an on-disk path.

        Prefers the canonical layout (image directly under the split dir) and
        falls back to an ``images/`` subdirectory for backward compatibility.
        """
        direct = self.split_dir / file_name
        if direct.exists():
            return direct
        nested = self.images_dir / file_name
        if nested.exists():
            return nested
        raise FileNotFoundError(
            f"Image '{file_name}' not found under {self.split_dir} (checked {direct} and {nested})."
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a COCO detection sample.

        Returns:
            dict with keys:
                - image: PIL Image
                - boxes: List of [x1, y1, x2, y2] boxes (COCO bbox is [x, y, w, h])
                - labels: List of class IDs (0-indexed)
                - sample_id: Image filename
        """
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]

        image_path = self._resolve_image_path(image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        annotations = self.annotations_by_image.get(image_id, [])

        boxes = []
        labels = []

        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h

            boxes.append([x1, y1, x2, y2])

            # Category ID to 0-indexed label
            # COCO categories are 1-indexed, so subtract 1
            category_id = ann["category_id"]
            label = category_id - 1  # Convert to 0-indexed
            labels.append(label)

        return {
            "image": image,
            "boxes": boxes,  # List of [x1, y1, x2, y2]
            "labels": labels,  # List of class IDs (0-indexed)
            "sample_id": image_info["file_name"],
        }


def export_to_coco(
    dataset: Dataset,
    split_dir: str | Path,
    num_classes: int,
    class_names: list[str] | None = None,
) -> Path:
    """Export an in-memory detection dataset to RF-DETR (Roboflow) COCO format.

    RF-DETR's ``build_roboflow`` expects, per split directory::

        split_dir/
        ├── _annotations.coco.json
        └── <image>.jpg ...

    The dataset is expected to yield dicts of the form produced by
    :class:`RandomDetectionDataset`::

        {"image": PIL.Image, "boxes": [[x1, y1, x2, y2], ...],
         "labels": [int, ...]  # 0-indexed, "sample_id": str}

    Format conversions applied here:
        - boxes ``[x1, y1, x2, y2]`` -> COCO ``[x, y, w, h]``
        - labels 0-indexed -> ``category_id`` 1-indexed (id 0 is the Roboflow
          "objects" placeholder whose ``supercategory`` is ``"none"``; RF-DETR
          counts it toward ``num_classes`` but it never appears in annotations)

    Args:
        dataset: Dataset yielding the standard detection sample dict.
        split_dir: Output directory for this split.
        num_classes: Number of real object classes.
        class_names: Optional names for the real classes (length ``num_classes``).

    Returns:
        The split directory path.
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Roboflow convention: a leading placeholder category with supercategory "none".
    categories = [{"id": 0, "name": "objects", "supercategory": "none"}]
    categories += [{"id": i + 1, "name": class_names[i], "supercategory": "objects"} for i in range(num_classes)]

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"]
        # Use the stem so an existing extension (e.g. "image1.jpg") is not
        # duplicated into "image1.jpg.jpg".
        sample_id = sample.get("sample_id", f"img_{idx:05d}")
        file_name = f"{Path(sample_id).stem}.jpg"

        # JPEG has no alpha channel; ensure RGB so non-RGB inputs don't error.
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(split_dir / file_name, format="JPEG")
        width, height = image.size
        images.append({"id": idx, "file_name": file_name, "width": width, "height": height})

        for box, label in zip(sample["boxes"], sample["labels"]):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": int(label) + 1,  # 0-indexed -> 1-indexed
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_dict = {
        "info": {"description": "Auto-generated dummy detection dataset"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    with open(split_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco_dict, f)

    return split_dir


def create_dummy_coco_dataset(
    root: str | Path,
    num_samples: int = 100,
    num_classes: int = 6,
    image_size: int = 640,
    max_boxes_per_image: int = 10,
    seed: int = 42,
    class_names: list[str] | None = None,
) -> Path:
    """Generate a Roboflow-format COCO dataset on disk for RF-DETR training.

    Creates ``train``/``valid``/``test`` splits (all three are required by
    RF-DETR, which builds every split unconditionally). Splits use distinct but
    deterministic seeds so the data is reproducible for a given ``seed``.

    If the dataset already exists (``train/_annotations.coco.json`` present), it
    is reused as a cache and not regenerated.

    Args:
        root: Root directory to write the dataset into.
        num_samples: Number of training samples (valid/test scale down from it).
        num_classes: Number of real object classes.
        image_size: Square image side length in pixels.
        max_boxes_per_image: Maximum boxes per image.
        seed: Base random seed (train=seed, valid=seed+1, test=seed+2).
        class_names: Optional class names.

    Returns:
        The dataset root directory (pass this as RF-DETR's ``dataset_dir``).
    """
    root = Path(root)

    # Cache: reuse if already generated for this exact configuration.
    if (root / "train" / "_annotations.coco.json").exists():
        return root

    val_samples = max(2, num_samples // 5)
    splits = {
        "train": (num_samples, seed),
        "valid": (val_samples, seed + 1),
        "test": (val_samples, seed + 2),
    }

    for split, (n, split_seed) in splits.items():
        ds = RandomDetectionDataset(
            num_samples=n,
            num_classes=num_classes,
            image_size=image_size,
            max_boxes_per_image=max_boxes_per_image,
            seed=split_seed,
        )
        export_to_coco(ds, root / split, num_classes=num_classes, class_names=class_names)

    return root


# Registry functions
@register_dataset("random_detection", aliases=["random", "dummy"])
def build_random_detection_dataset(
    num_samples: int = 100,
    num_classes: int = 6,
    image_size: int = 640,
    max_boxes_per_image: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> RandomDetectionDataset:
    """Build a random detection dataset for testing.

    Args:
        num_samples: Number of samples
        num_classes: Number of detection classes
        image_size: Image size (square)
        max_boxes_per_image: Maximum boxes per image
        seed: Random seed
        **kwargs: Ignored additional parameters

    Returns:
        RandomDetectionDataset instance
    """
    return RandomDetectionDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        max_boxes_per_image=max_boxes_per_image,
        seed=seed,
    )


@register_dataset("coco_detection", aliases=["coco"])
def build_coco_detection_dataset(
    coco_root: str,
    split: str = "train",
    transform: Any = None,
    **kwargs: Any,
) -> COCODetectionDataset:
    """Build a COCO detection dataset.

    Args:
        coco_root: Root directory of COCO dataset
        split: Dataset split ('train', 'valid', 'test')
        transform: Optional image transforms
        **kwargs: Ignored additional parameters

    Returns:
        COCODetectionDataset instance
    """
    return COCODetectionDataset(
        coco_root=coco_root,
        split=split,
        transform=transform,
    )
