#!/usr/bin/env bash
# Thin wrapper; the logic lives in
# docker/prepare_models.py (it needs torch to strip the Lightning checkpoints).
set -e
ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
cd "$ROOT"
PYTHONPATH=. ${PYTHON:-uv run python} docker/prepare_models.py "$@"
