#!/usr/bin/env bash
# Build both submission images (thin wrapper over the per-task do_build.sh, which
# is what the official template calls).
#   bash docker/build.sh            # task1 + task2
#   bash docker/build.sh task2      # one of them
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
for t in ${1:-task1 task2}; do
  echo "== building topaneu-26-$t =="
  bash "${SCRIPT_DIR}/${t}/do_build.sh"
done
echo "done. Export for upload with: bash docker/task2/do_save.sh"
