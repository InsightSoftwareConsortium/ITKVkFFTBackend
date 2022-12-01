#!/bin/bash

# This module should be pulled and run from an ITKModule root directory to generate the Linux python wheels of this module,
# it is used by the circle.yml file contained in ITKModuleTemplate: https://github.com/InsightSoftwareConsortium/ITKModuleTemplate
#
# Forked from
# https://github.com/InsightSoftwareConsortium/ITKPythonPackage/blob/master/scripts/dockcross-manylinux-download-cache-and-build-module-wheels.sh

# -----------------------------------------------------------------------

# Verifies that unzstd binary is available to decompress ITK build archives.
unzstd_exe=`(which unzstd)`

if [[ -z ${unzstd_exe} ]]; then
  echo "ERROR: can not find required binary 'unzstd' "
  exit 255
fi

# Expect unzstd > v1.3.2, see discussion in `dockcross-manylinux-build-tarball.sh`
${unzstd_exe} --version

# -----------------------------------------------------------------------
# Fetch build archive

TARBALL_SPECIALIZATION="-manylinux${MANYLINUX_VERSION:=_2_28}"
TARBALL_NAME="ITKPythonBuilds-linux${TARBALL_SPECIALIZATION}.tar"
curl -L https://github.com/InsightSoftwareConsortium/ITKPythonBuilds/releases/download/${ITK_PACKAGE_VERSION:=v5.3.0}/${TARBALL_NAME}.zst -O

${unzstd_exe} --long=31 ${TARBALL_NAME}.zst -o ${TARBALL_NAME}
if [ "$#" -lt 1 ]; then
  echo "Extracting all files";
  tar xf ${TARBALL_NAME}
else
  echo "Extracting files relevant for: $1";
  tar xf ${TARBALL_NAME} ITKPythonPackage/scripts/
  tar xf ${TARBALL_NAME} ITKPythonPackage/ITK-source/
  tar xf ${TARBALL_NAME} ITKPythonPackage/oneTBB-prefix/
  tar xf ${TARBALL_NAME} --wildcards ITKPythonPackage/ITK-$1*
fi
rm ${TARBALL_NAME}

# Optional: Update build scripts
if [[ -n ${ITKPYTHONPACKAGE_TAG} ]]; then
  echo "Updating build scripts to ${ITKPYTHONPACKAGE_ORG:=InsightSoftwareConsortium}/ITKPythonPackage@${ITKPYTHONPACKAGE_TAG}"
  git clone "https://github.com/${ITKPYTHONPACKAGE_ORG}/ITKPythonPackage.git" "IPP-tmp"
  pushd IPP-tmp/
  git checkout "${ITKPYTHONPACKAGE_TAG}"
  git status
  popd

  rm -rf ITKPythonPackage/scripts/
  cp -r IPP-tmp/scripts ITKPythonPackage/
  rm -rf IPP-tmp/
fi

cp -a ITKPythonPackage/oneTBB-prefix ./
