#!/usr/bin/env python
"""Build the local test fixtures do_test_run.sh feeds to the containers.

Grand Challenge hands the algorithm an .mha per case plus an inputs.json naming
the socket; this reproduces that layout from the released .nii.gz volumes, so a
local run exercises the same reader path (SimpleITK, not nibabel) as the real
job:

    docker/test/input/interf0/inputs.json
    docker/test/input/interf0/images/head-ct-angio/<case>.mha     # CTA
    docker/test/input/interf1/inputs.json
    docker/test/input/interf1/images/head-mr-angio/<case>.mha     # MRA

Usage:
    PYTHONPATH=. uv run python docker/make_test_input.py
    PYTHONPATH=. uv run python docker/make_test_input.py --ct <case> --mr <case>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
DEP = Path("data/raw/topaneu")   # same default as scripts/run_pipeline.sh (RAW)

SOCKETS = {
    "interf0": ("head-ct-angiography", "images/head-ct-angio"),
    "interf1": ("head-mr-angiography", "images/head-mr-angio"),
}


def inputs_json(slug: str, relative_path: str) -> list:
    """The subset of the GC-generated inputs.json that main.py actually reads."""
    return [{
        "socket": {
            "slug": slug,
            "relative_path": relative_path,
            "is_image_kind": True,
            "is_panimg_kind": True,
            "is_dicom_image_kind": False,
            "is_json_kind": False,
            "is_file_kind": False,
        },
        "file": None,
        "image": {"name": "the_original_filename_of_the_file_that_was_uploaded_or_image_set"},
        "value": None,
    }]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, default=DEP / "images")
    ap.add_argument("--out", type=Path, default=HERE / "test/input")
    ap.add_argument("--ct", default="topaneu_center2_ct_107", help="case id for interf0 (fold0-val, GT location 26)")
    ap.add_argument("--mr", default="topaneu_center1_mr_113", help="case id for interf1 (fold0-val, GT location 36)")
    args = ap.parse_args()

    for interf, case in (("interf0", args.ct), ("interf1", args.mr)):
        slug, rel = SOCKETS[interf]
        src = args.images_dir / f"{case}_0000.nii.gz"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        dst_dir = args.out / interf / rel
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True)
        img = sitk.ReadImage(str(src))
        out = dst_dir / f"{case}.mha"
        sitk.WriteImage(img, str(out), useCompression=True)
        (args.out / interf / "inputs.json").write_text(json.dumps(inputs_json(slug, rel), indent=4))
        print(f"{interf}: {case}  size={img.GetSize()} spacing={img.GetSpacing()} "
              f"-> {out} ({out.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
