"""Task2 (vessel-specific segmentation) inference.

Grand Challenge template shim: `main.py` hands us the SimpleITK image, we return
a 1-channel uint8 mask with the vessel-specific location label (0-52) at each
aneurysm voxel, which main.py writes to
/output/images/aneurysm-segmentation/output.mha.

The returned image copies the input's origin/spacing/direction, so its shape and
geometry match the input exactly (the evaluator requires `pred.shape == gt.shape`;
the stock template drops the geometry). All the work happens in
`topaneu_infer.predict`, shared with Task1.
"""

import SimpleITK as sitk

from topaneu_infer import predict


def infer_ct(img: sitk.Image) -> sitk.Image:
    """Segments aneurysms in CTA images."""
    _, mask = predict(img, "CTA")
    return mask


def infer_mr(img: sitk.Image) -> sitk.Image:
    """Segments aneurysms in MRA images."""
    _, mask = predict(img, "MRA")
    return mask
