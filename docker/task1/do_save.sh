#!/usr/bin/env bash
# Rebuild, then export the image (and the shared model tarball) for upload to
# Grand Challenge. Mirrors the official template's do_save.sh.
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_DIR=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
DOCKER_IMAGE_TAG="topaneu-26-task1"

echo ""
echo "= STEP 1 = (Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"
echo "==== Done"
echo ""

build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi
formatted_build_info=$(echo $build_timestamp | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')
output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/$output_filename"

echo "= STEP 2 = Saving the image"
echo "This can take a while."
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_path"
printf "Saved as: \e[32m${output_filename}\e[0m\n"
echo "==== Done"
echo ""

echo "= STEP 3 = Packing the model"
if [ ! -d "${DOCKER_DIR}/model" ]; then
    echo "Error: ${DOCKER_DIR}/model is missing — run 'bash docker/prepare_models.sh' first"
    exit 1
fi
tar -czf "${DOCKER_DIR}/model.tar.gz" -C "${DOCKER_DIR}/model" .
printf "Saved as: \e[32mdocker/model.tar.gz\e[0m\n"
echo "==== Done"
echo ""
printf "\e[31mIMPORTANT: upload model.tar.gz as a separate Model on your Algorithm.\e[0m\n"
printf "Task1 and Task2 share the SAME model.tar.gz.\n"
