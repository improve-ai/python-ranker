import os
from setuptools import Extension, find_packages, setup

from Cython.Build import cythonize
import numpy as np

CYTHON_MODULE_DIR = 'cythonized_feature_encoding'
EXTENSION = 'pyx'
IMPROVE_DIR = 'improveai'


if __name__ == '__main__':

    cython_feature_encoding_utils_path_str = \
        os.sep.join(
            [IMPROVE_DIR, CYTHON_MODULE_DIR, 'cythonized_feature_encoding_utils.{}'.format(EXTENSION)])

    cython_feature_encoding_utils_ext = \
        Extension(
            '{}.{}.cythonized_feature_encoding_utils'.format(IMPROVE_DIR, CYTHON_MODULE_DIR),
            sources=[cython_feature_encoding_utils_path_str],
            include_dirs=[np.get_include(), os.sep.join(['.', IMPROVE_DIR, CYTHON_MODULE_DIR])])

    cython_feature_encoder_path_str = \
        os.sep.join(
            [IMPROVE_DIR, CYTHON_MODULE_DIR, 'cythonized_feature_encoder.{}'.format(EXTENSION)])

    cython_feature_encoder_ext = \
        Extension(
            '{}.{}.cythonized_feature_encoder'.format(IMPROVE_DIR, CYTHON_MODULE_DIR),
            sources=[cython_feature_encoder_path_str],
            include_dirs=[np.get_include(), os.sep.join(['.', IMPROVE_DIR, CYTHON_MODULE_DIR])])

    # TODO some requirements may depend on gcc version, e.g. coremltools
    install_requires = [
        "setuptools", "wheel",
        "Cython>=0.29.14",
        "xxhash==2.0.0",
        'coremltools==4.1',
        "requests>=2.24.0",
        "numpy>=1.19.2",
        "xgboost==1.4.2",
        "simplejson==3.17.2",
        "orjson",
        "svix-ksuid"]

    setup(
        name='improveai',
        version='7.0.1',
        description='Improve AI: AI Decisions for iOS, Android, and the Cloud',
        author='Justin Chapweske',
        author_email='support@improve.ai',
        url='https://github.com/improve-ai/python-sdk',
        packages=find_packages(exclude=['*.tox*', '*tests*']),
        install_requires=install_requires,
        ext_modules=cythonize([cython_feature_encoding_utils_ext, cython_feature_encoder_ext], language_level="3"),
        include_dirs=[np.get_include(), '.'],
        include_package_data=True,
        zip_safe=False)
