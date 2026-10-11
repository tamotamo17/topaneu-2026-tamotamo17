#!/usr/bin/env bash
# Build and run the Task2 container exactly as Grand Challenge would
# (--network none, --gpus all, /tmp on a throwaway volume, weights read-only at
# /opt/ml/model). Mirrors the official template's do_test_run.sh; the test
# fixtures and the model dir are shared between the two tasks, so they live in
# docker/ rather than in this directory.
#
# Prerequisites:
#   bash docker/prepare_models.sh     # -> docker/model/
#   PYTHONPATH=. uv run python docker/make_test_input.py   # -> docker/test/input/
#
# On a machine without a GPU, drop '--gpus all' from the run command below.
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_DIR=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
DOCKER_IMAGE_TAG="topaneu-26-task2"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${DOCKER_DIR}/test/input"
OUTPUT_DIR="${DOCKER_DIR}/test/output/task2"
MODEL_DIR="${DOCKER_DIR}/model"

for d in "$INPUT_DIR" "$MODEL_DIR"; do
    [ -d "$d" ] || { echo "Error: $d is missing — see the header of this script"; exit 1; }
done

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    docker run --rm --platform=linux/amd64 --quiet \
      --volume "$OUTPUT_DIR":/output --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG -c "chmod -R -f o+rwX /output/* || true"
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

chmod -R -f o+rX "$INPUT_DIR" "$MODEL_DIR"

for interf in interf0 interf1; do
  if [ -d "${OUTPUT_DIR}/${interf}" ]; then
    chmod -f o+rwX "${OUTPUT_DIR}/${interf}"
    echo "=+= Cleaning up any earlier output in ${interf}"
    docker run --rm --platform=linux/amd64 --quiet \
        --volume "${OUTPUT_DIR}/${interf}":/output --entrypoint /bin/sh \
        $DOCKER_IMAGE_TAG -c "rm -rf /output/* || true"
  else
    mkdir -p -m o+rwX "${OUTPUT_DIR}/${interf}"
  fi
done

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null
trap cleanup EXIT

run_docker_forward_pass() {
    local interface_dir="$1"
    echo "=+= Doing a forward pass on ${interface_dir}"
    docker run --rm --gpus all \
        --platform=linux/amd64 \
        --network none \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        --volume "${MODEL_DIR}":/opt/ml/model:ro \
        "$DOCKER_IMAGE_TAG"
    echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}

run_docker_forward_pass "interf0"   # CTA
run_docker_forward_pass "interf1"   # MRA

echo "=+= Save this image for uploading via ./do_save.sh"
