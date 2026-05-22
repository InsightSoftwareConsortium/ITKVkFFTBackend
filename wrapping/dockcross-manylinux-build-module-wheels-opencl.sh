#!/bin/bash

# Run this script to build the Python wheel packages for Linux for an ITK
# external module.
#
# Versions can be restricted by passing them in as arguments to the script
# For example,
#
#   scripts/dockcross-manylinux-build-module-wheels.sh cp39
#
# Forked from
# https://github.com/InsightSoftwareConsortium/ITKPythonPackage/blob/master/scripts/dockcross-manylinux-build-module-wheels.sh

# Generate dockcross scripts

MANYLINUX_VERSION=${MANYLINUX_VERSION:=_2_28}
# Match ITKPythonPackage v6.0b02's dockcross image, which ships a recent
# enough `python -m build` to accept --verbose (the older 20221108 image
# fails with "unrecognized arguments: --verbose").
IMAGE_TAG=${IMAGE_TAG:=20260203-3dfb3ff}
OPENCL_ICD_LOADER_TAG=${OPENCL_ICD_LOADER_TAG:=v2025.07.22}
OPENCL_HEADERS_TAG=${OPENCL_HEADERS_TAG:=v2025.07.22}

docker run --rm dockcross/manylinux${MANYLINUX_VERSION}-x64:${IMAGE_TAG} > /tmp/dockcross-manylinux-x64
chmod u+x /tmp/dockcross-manylinux-x64

script_dir=$(cd $(dirname $0) || exit 1; pwd)

mkdir -p $(pwd)/tools
chmod 777 $(pwd)/tools
# Build wheels
mkdir -p dist

# Build OpenCL-ICD-Loader before ITKVkFFTBackend
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
#
# Resolve the actual versioned libOpenCL.so filename produced by the
# OpenCL-ICD-Loader build. v2021.04.29 produced libOpenCL.so.1.2; the
# v2025.07.22 build produces libOpenCL.so.1.0.0. Bind-mount whichever the
# loader actually wrote.
OPENCL_SO=$(ls $(pwd)/OpenCL-ICD-Loader-build/libOpenCL.so.1.* 2>/dev/null | head -1)
if [[ -z "${OPENCL_SO}" ]]; then
  echo "ERROR: could not find OpenCL-ICD-Loader-build/libOpenCL.so.1.* — did the loader build?" >&2
  exit 1
fi
echo "Using OpenCL loader: ${OPENCL_SO}"

DOCKER_ARGS="-v $(pwd)/dist:/work/dist/ -v $script_dir/../ITKPythonPackage:/ITKPythonPackage -v $(pwd)/tools:/tools"
DOCKER_ARGS+=" -v $(pwd)/OpenCL-ICD-Loader/inc/CL:/usr/include/CL"
DOCKER_ARGS+=" -v ${OPENCL_SO}:/usr/lib64/libOpenCL.so.1"
DOCKER_ARGS+=" -v ${OPENCL_SO}:/usr/lib64/libOpenCL.so"
DOCKER_ARGS+=" -e MANYLINUX_VERSION"

/tmp/dockcross-manylinux-x64 \
  -a "$DOCKER_ARGS" \
  "/ITKPythonPackage/scripts/internal/manylinux-build-module-wheels.sh" "$@"
