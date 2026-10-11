#!/usr/bin/env bash
# Build the Task1 submission image as topaneu-26-task1:latest.
# Same contract as the official template's do_build.sh, except the build CONTEXT
# is the repository root (the image needs src/, scripts/, configs/).
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT=$( cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd )
DOCKER_IMAGE_TAG="topaneu-26-task1"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  --file "${SCRIPT_DIR}/Dockerfile" \
  "$ROOT" 2>&1
