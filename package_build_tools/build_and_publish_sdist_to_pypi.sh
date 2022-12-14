#!/bin/bash
PYPI_PACKAGE_NAME=${1:-'improveai'}
MAJOR_VERSION=${2:-'7.2'}
MINOR_VERSION=${3:-'2'}

echo "Building ${PYPI_PACKAGE_NAME} -> version ${MAJOR_VERSION}.${MINOR_VERSION}"
# cleanup previous build
rm -rf .tox/ dist/ improveai_test.egg-info/ improveai/cythonized_feature_encoding/cythonized_feature_encoder.c* improveai/cythonized_feature_encoding/cythonized_feature_encoding_utils.c* "${PYPI_PACKAGE_NAME}.egg*"

pip3 install --upgrade pip wheel build twine
# compile *.c cython files (python will look for *.pyx files to import)
python3 -m build

# upload to pypi
echo "Uploading dist/${PYPI_PACKAGE_NAME}-${MAJOR_VERSION}.${MINOR_VERSION}.tar.gz to pypi"
python3 -m twine upload --verbose --repository pypi "dist/${PYPI_PACKAGE_NAME}-${MAJOR_VERSION}.${MINOR_VERSION}.tar.gz"