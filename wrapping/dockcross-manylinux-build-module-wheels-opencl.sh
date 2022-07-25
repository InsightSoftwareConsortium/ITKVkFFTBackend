#!/bin/bash

# Run this script to build the Python wheel packages for Linux for an ITK
# external module.
#
# Versions can be restricted by passing them in as arguments to the script
# For example,
#
#   scripts/dockcross-manylinux-build-module-wheels.sh cp39

# Generate dockcross scripts

MANYLINUX_VERSION="_2_28"
IMAGE_TAG=20220715-9ce3707
OPENCL_ICD_LOADER_TAG=v2021.04.29
OPENCL_HEADERS_TAG=v2021.04.29

docker run --rm dockcross/manylinux${MANYLINUX_VERSION}-x64:${IMAGE_TAG} > /tmp/dockcross-manylinux-x64
chmod u+x /tmp/dockcross-manylinux-x64

script_dir=$(cd $(dirname $0) || exit 1; pwd)

if ! test -d ./OpenCL-ICD-Loader; then
  git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
  pushd OpenCL-ICD-Loader
  git checkout ${OPENCL_ICD_LOADER_TAG}
  popd
  pushd OpenCL-ICD-Loader/inc
  git clone https://github.com/KhronosGroup/OpenCL-Headers
  pushd OpenCL-Headers
  git checkout ${OPENCL_HEADERS_TAG}
  popd
  cp -r OpenCL-Headers/CL ./
  popd
  mkdir OpenCL-ICD-Loader-build
  /tmp/dockcross-manylinux-x64 cmake -BOpenCL-ICD-Loader-build -HOpenCL-ICD-Loader -GNinja
  /tmp/dockcross-manylinux-x64 ninja -COpenCL-ICD-Loader-build

  rm -rf ITKPythonPackage/standalone-x64-build/ITKs/Modules/Core/GPUCommon/
fi

# Build wheels
mkdir -p dist
DOCKER_ARGS="-v $(pwd)/dist:/work/dist/ -v $script_dir/../ITKPythonPackage:/ITKPythonPackage -v $(pwd)/tools:/tools -v $(pwd)/OpenCL-ICD-Loader/inc/CL:/usr/include/CL -v $(pwd)/OpenCL-ICD-Loader-build/libOpenCL.so.1.2:/usr/lib64/libOpenCL.so.1 -v $(pwd)/OpenCL-ICD-Loader-build/libOpenCL.so.1.2:/usr/lib64/libOpenCL.so"
/tmp/dockcross-manylinux-x64 \
  -a "$DOCKER_ARGS" \
  "/ITKPythonPackage/scripts/internal/manylinux-build-module-wheels.sh" "$@"
