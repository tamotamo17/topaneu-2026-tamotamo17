"""Grand Challenge entrypoint for Task1 (vessel-specific classification).

Follows the official TopAneu-26 template (templates/task1/main.py): the socket
slugs in /input/inputs.json select the modality handler, the handler reads the
image with SimpleITK and writes the detected location list to
/output/detected-aneurysm-locations.json as a plain JSON array, e.g. [34, 35].

Added on top of the template: the written output is read back and checked
against the challenge's schema (a flat list of ints in 1-52) before the job
exits, so a malformed submission is visible in the job log rather than only in
the evaluator.
"""

import glob
import json
import os
from pathlib import Path

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
    write_json_file(location=OUTPUT_PATH / "detected-aneurysm-locations.json",
                    content=infer_ct(image))
    return 0


def interf1_handler():
    image = load_image_file(location=INPUT_PATH / "images/head-mr-angio")
    write_json_file(location=OUTPUT_PATH / "detected-aneurysm-locations.json",
                    content=infer_mr(image))
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


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))
    check_output(location)


def check_output(location):
    """Read back the written file and assert the challenge's output contract."""
    value = load_json_file(location=location)
    ok = (isinstance(value, list)
          and all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 52
                  for v in value))
    print(f"[topaneu] wrote {location}: {value} (schema {'OK' if ok else 'INVALID'})",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(run())
