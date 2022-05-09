import os
from setuptools import find_packages, setup


def gcc_and_py3_dev_installed():
    gcc_installed = os.system('gcc -v > /dev/null 2>&1') == 0
    py3_dev_installed = os.system('python3-config --help > /dev/null 2>&1') == 0
    return gcc_installed and py3_dev_installed


CYTHON_MODULE_DIR = 'cythonized_feature_encoding'
EXTENSION = 'pyx'
IMPROVE_DIR = 'improveai'


if __name__ == '__main__':

    install_requires = [
        "Cython>=0.29.14",
        "xxhash",
        "requests>=2.24.0",
        "numpy",
        "xgboost==1.4.2",
        "orjson",
        "svix-ksuid"]

    setup_kwargs = {
        'packages': find_packages(exclude=['*.tox*', '*tests*']),
        'install_requires': install_requires,
        'include_package_data': True}

    if gcc_and_py3_dev_installed():
        from Cython.Build import cythonize
        import numpy as np
        from setuptools import Extension

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

        # append setup kwargs with cythonization info
        setup_kwargs.update(
            {'ext_modules': cythonize([cython_feature_encoding_utils_ext, cython_feature_encoder_ext], language_level="3"),
             'include_dirs': [np.get_include(), '.']
             })

    setup(**setup_kwargs)
