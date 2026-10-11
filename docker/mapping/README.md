# Label mappings (baked into the containers)

`scripts/anatomical_label.py` reads `vessel_mapping.json` (36 vessel classes) and
`location_mapping.json` (52 location classes) from `$TOPANEU_DEPLOYMENT` at
import time. The data release is not inside the container, so only these two
files are baked into the image and the Dockerfile sets
`TOPANEU_DEPLOYMENT=/opt/app/mapping`. Outside the container the same directory
is the default (`docker/mapping/`).

Source: the 2026-08 data release (417 cases / 52 location classes). The two files
are **byte-identical** to those of the 2026-08-17 revision, so location ids are
the same whichever of the two releases is used. They are NOT compatible with the
first release (50 classes), which assigns ids 47-52 differently — shipping the
old mapping produces ids that look valid but are all shifted.

To update:
```bash
cp <release>/{location_mapping.json,vessel_mapping.json} docker/mapping/
```
