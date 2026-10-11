"""Grand Challenge entrypoint for Task2 (vessel-specific segmentation).

Follows the official TopAneu-26 template (templates/task2/main.py): the socket
slugs in /input/inputs.json select the modality handler, the handler reads the
image with SimpleITK and writes the predicted mask to
/output/images/aneurysm-segmentation/output.mha.

Added on top of the template: the written .mha is read back and checked against
the challenge's output contract (uint8, values in 0-52, same shape and geometry
as the input) before the job exits, so a malformed submission is visible in the
job log rather than only in the evaluator.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK

from inference import infer_ct, infer_mr

# Grand Challenge mounts these; the env overrides are for the local no-docker
# run described in docker/README.md and are unset on the platform.
INPUT_PATH = Path(os.environ.get("TOPANEU_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("TOPANEU_OUTPUT", "/output"))


def run():
    interface_key = get_interface_key()
    handler = {
        ("head-ct-angiography",): interf0_handler,
        ("head-mr-angiography",): interf1_handler,
    }[interface_key]
    return handler()


def interf0_handler():
    image = load_image_file(location=INPUT_PATH / "images/head-ct-angio")
    write_image_file(location=OUTPUT_PATH / "images/aneurysm-segmentation",
                     image=infer_ct(image), reference=image)
    return 0


def interf1_handler():
    image = load_image_file(location=INPUT_PATH / "images/head-mr-angio")
    write_image_file(location=OUTPUT_PATH / "images/aneurysm-segmentation",
                     image=infer_mr(image), reference=image)
    return 0


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    print(f"[topaneu] interface sockets: {sorted(socket_slugs)}", flush=True)
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def load_image_file(*, location):
    input_files = (
        glob.glob(str(location / "*.tif"))
        + glob.glob(str(location / "*.tiff"))
        + glob.glob(str(location / "*.mha"))
    )
    print(f"[topaneu] reading {input_files[0]}", flush=True)
    return SimpleITK.ReadImage(input_files[0])


def write_image_file(*, location, image, reference):
    location.mkdir(parents=True, exist_ok=True)
    path = location / "output.mha"
    SimpleITK.WriteImage(image, str(path), useCompression=True)
    check_output(path, reference)


def check_output(path, reference):
    """Read the written .mha back and assert the challenge's output contract."""
    written = SimpleITK.ReadImage(str(path))
    arr = SimpleITK.GetArrayFromImage(written)
    ref = SimpleITK.GetArrayFromImage(reference)
    problems = []
    if arr.dtype != np.uint8:
        problems.append(f"dtype {arr.dtype} != uint8")
    if arr.shape != ref.shape:
        problems.append(f"shape {arr.shape} != input {ref.shape}")
    if arr.size and int(arr.max()) > 52:
        problems.append(f"label {int(arr.max())} out of range 0-52")
    if written.GetSpacing() != reference.GetSpacing():
        problems.append(f"spacing {written.GetSpacing()} != input {reference.GetSpacing()}")
    labels = np.unique(arr[arr > 0]).tolist()
    print(f"[topaneu] wrote {path}: shape={arr.shape} dtype={arr.dtype} "
          f"labels={labels} fg_voxels={int((arr > 0).sum())} "
          f"({'OK' if not problems else 'INVALID: ' + '; '.join(problems)})", flush=True)


if __name__ == "__main__":
    raise SystemExit(run())
