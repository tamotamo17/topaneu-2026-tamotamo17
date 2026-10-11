"""Modality-aware 2D projection for vessel-ROI detection (MRA vs CTA).

Both MRA and CTA depict vessels, but they need **different** 2D encodings:

- **MRA** (bright-blood): vessels are the brightest voxels, so a maximum-
  intensity projection (MIP) over the projection axis renders the whole vessel
  tree cleanly. Intensity is scaled by percentile (1-99).
- **CTA**: bone (HU up to ~2000+) is far brighter than contrast-enhanced
  vessels (~100-500 HU), so a full-depth MIP is dominated by the skull and
  vessels wash out. We instead use a **narrow vessel window [100, 500] HU**
  (brain ~35 HU -> black, contrast vessels -> bright) and MIP over a **central
  slab** (``±CTA_SLAB_HALF`` voxels) rather than the whole depth. The slab keeps
  the intracranial vasculature (Circle of Willis + major branches, which sit
  near mid-brain) while excluding most of the skull convexity, so the vessel
  tree renders cleanly — the CTA analogue of the MRA MIP. (A single central
  slice, or the wider [-100,600] window, left the vessels faint; a full-depth
  MIP saturated to bone — both were rejected empirically.)

The detection target box is always derived from the *mask* MIP (full 3D extent);
only the *image* encoding differs by modality — see
``scripts/make_vessel_roi_detection_data.py``.

Plane convention (arrays are ``[X, Y, Z]``):
    - ``"cor"`` (coronal): reduce/slice axis 1 (Y) -> image ``[X, Z]``
    - ``"sag"`` (sagittal): reduce/slice axis 0 (X) -> image ``[Y, Z]``
"""

from __future__ import annotations

import numpy as np

# CTA: narrow vessel window (HU) + central-slab MIP half-thickness (voxels).
CTA_HU_WINDOW = (100.0, 500.0)
CTA_SLAB_HALF = 30
MRA_PCT = (1.0, 99.0)
_PLANE_AXIS = {"cor": 1, "sag": 0}


def modality_of(case_id: str) -> str:
    """Infer modality from the case id (``..._ct_...`` -> CTA, else MRA)."""
    return "CTA" if "_ct_" in case_id else "MRA"


def value_normalize(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to [lo, hi] and scale to uint8 [0, 255] (CTA HU windowing)."""
    out = np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def pct_normalize(img: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """Clip to the [p0, p1] percentiles and scale to uint8 [0, 255] (MRA)."""
    lo, hi = np.percentile(img, p0), np.percentile(img, p1)
    out = np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def make_projection(
    image3d: np.ndarray,
    plane: str,
    modality: str,
    cta_window: tuple[float, float] = CTA_HU_WINDOW,
    cta_slab: int = CTA_SLAB_HALF,
) -> np.ndarray:
    """Return the modality-appropriate 2D uint8 projection for ``plane``.

    Args:
        image3d: 3D volume ``[X, Y, Z]`` (raw intensity / HU).
        plane: ``"cor"`` or ``"sag"``.
        modality: ``"CTA"`` or ``"MRA"``.
        cta_window: CTA HU window (lo, hi). CTA only.
        cta_slab: CTA central-slab half-thickness (voxels). ``0`` = single
            central slice; ``N>0`` = MIP over the central ``±N`` slab. CTA only.
    """
    if plane not in _PLANE_AXIS:
        raise ValueError(f"plane must be 'cor' or 'sag', got {plane!r}")
    axis = _PLANE_AXIS[plane]

    if modality == "CTA":
        # Vessel window over a central slab, then MIP (slab=0 -> single slice):
        # bright vessel tree on a black brain, skull convexity excluded by slab.
        n = image3d.shape[axis]
        c = n // 2
        sl = [slice(None)] * image3d.ndim
        sl[axis] = slice(max(0, c - cta_slab), min(n, c + cta_slab + 1))
        windowed = value_normalize(image3d[tuple(sl)], *cta_window)
        return windowed.max(axis=axis)  # slab MIP (or single slice if slab=0)
    # MRA (and other MR): MIP first, then normalize the 2D projection by
    # percentile (percentiles of the MIP itself, not of the whole volume — the
    # latter over-saturates because every MIP pixel is a per-ray maximum).
    return pct_normalize(image3d.max(axis=axis), *MRA_PCT)
