"""Task1 (vessel-specific classification) inference.

Grand Challenge template shim: `main.py` hands us the SimpleITK image, we return
the list of detected location ids (1-52), which main.py writes as a plain JSON
array to /output/detected-aneurysm-locations.json — e.g. `[34, 35]`.

All the work happens in `topaneu_infer.predict`, shared with Task2, so the two
containers report the same findings.
"""

import SimpleITK as sitk

from topaneu_infer import predict


def infer_ct(img: sitk.Image) -> list:
    """Predicts aneurysm locations in CTA images."""
    locs, _ = predict(img, "CTA")
    return locs


def infer_mr(img: sitk.Image) -> list:
    """Predicts aneurysm locations in MRA images."""
    locs, _ = predict(img, "MRA")
    return locs
