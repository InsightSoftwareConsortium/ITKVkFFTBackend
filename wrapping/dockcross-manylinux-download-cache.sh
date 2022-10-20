#!/bin/bash

# This module should be pulled and run from an ITKModule root directory to generate the Linux python wheels of this module,
# it is used by the circle.yml file contained in ITKModuleTemplate: https://github.com/InsightSoftwareConsortium/ITKModuleTemplate

curl https://data.kitware.com/api/v1/file/592dd8068d777f16d01e1a92/download -o zstd-1.2.0-linux.tar.gz
gunzip -d zstd-1.2.0-linux.tar.gz
tar xf zstd-1.2.0-linux.tar

TARBALL_NAME="ITKPythonBuilds-linux${TARBALL_SPECIALIZATION}.tar"
curl -L https://github.com/InsightSoftwareConsortium/ITKPythonBuilds/releases/download/${ITK_PACKAGE_VERSION:=v5.3rc04.post2}/${TARBALL_NAME}.zst -O
./zstd-1.2.0-linux/bin/unzstd ${TARBALL_NAME}.zst -o ${TARBALL_NAME}
echo "Extracting all files"
tar xf ${TARBALL_NAME}
rm ${TARBALL_NAME}

mkdir tools
curl https://data.kitware.com/api/v1/file/5c0aa4b18d777f2179dd0a71/download -o doxygen-1.8.11.linux.bin.tar.gz
tar -xvzf doxygen-1.8.11.linux.bin.tar.gz -C tools

cp -a ITKPythonPackage/oneTBB-prefix ./
