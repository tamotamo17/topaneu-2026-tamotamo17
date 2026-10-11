# TopAneu-26 Task1 submission image.
#
# Build from the REPOSITORY ROOT as context (the pipeline needs src/, scripts/
# and configs/, which sit above this directory — this is the one deviation from
# the official template, whose do_build.sh uses the template dir as context):
#   bash docker/task1/do_build.sh
#
# Base image is the one the official TopAneu-26 template uses, so torch and CUDA
# match what the organizers test against. torch 2.9 loads the state_dicts trained
# under torch 2.8 fine (we ship plain tensors, see docker/prepare_models.sh).
FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS topaneu_task1

# PYTHONUNBUFFERED: without it a killed job (timeout / OOM) loses its log tail,
#   which is exactly the log you need to debug it.
# TOPANEU_DEPLOYMENT: scripts/anatomical_label.py reads the vessel/location id
#   mappings from here at import time. The data release is not in the image,
#   only those two JSONs (docker/mapping/).
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: Grand Challenge runs with
#   --network none, so fail fast instead of retrying into the time limit.
# TOPANEU_STRICT=0: on a pipeline error, log the traceback and emit an
#   empty-but-valid output — a sanity-check phase learns more from a finished
#   job than a dead one. Set to 1 to make failures fatal.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app \
    TOPANEU_DEPLOYMENT=/opt/app/mapping \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOPANEU_STRICT=0

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# requirements.txt pins opencv-python-headless, but rfdetr's dependency tail
# (supervision/roboflow) pulls plain opencv-python in as well, and that build
# wins the shared cv2 package dir — so cv2 really does need libGL/libxcb here.
# Without them `import rfdetr` dies at Step1 with `ImportError: libxcb.so.1`,
# and TOPANEU_STRICT=0 turns that into a silent empty prediction. Only a real
# container run catches it: outside the image the host supplies these libs.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
USER user

COPY --chown=user:user docker/task1/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt

# Pipeline code + configs + label mappings. Weights are NOT baked in: they are
# uploaded separately as a Model and mounted at /opt/ml/model at runtime.
COPY --chown=user:user src/            /opt/app/src/
COPY --chown=user:user scripts/        /opt/app/scripts/
COPY --chown=user:user configs/        /opt/app/configs/
COPY --chown=user:user docker/mapping/ /opt/app/mapping/
COPY --chown=user:user docker/common/topaneu_infer.py /opt/app/topaneu_infer.py
COPY --chown=user:user docker/task1/inference.py     /opt/app/inference.py
COPY --chown=user:user docker/task1/main.py          /opt/app/main.py

# Grand Challenge de-duplicates uploads by checksum, so re-uploading a
# byte-identical image to a second Algorithm is rejected ("... with this
# checksum already exists"). Bumping this label changes the image config digest
# — a genuinely different image, not just a differently-compressed tarball of
# the same one — which is what lets the same build be uploaded again. Last
# layer on purpose: it is metadata only, so bumping it does not invalidate the
# pip-install layer above.
ARG SUBMISSION_VERSION=submission-v2
LABEL version="${SUBMISSION_VERSION}"

ENTRYPOINT ["python", "main.py"]
