from setuptools import Extension, find_packages, setup

import os
import pydoc


if __name__ == '__main__':

    installed_packages = [pkg.name for pkg in pydoc.pkgutil.iter_modules()]

    reqs_path = os.getenv('SDK_HOME_PATH', None)

    if not reqs_path:
        reqs_path = \
            os.sep.join(str(__file__).split(os.sep)[:-1] + ['requirements.txt'])

    with open(reqs_path) as rqf:
        install_reqs = \
            [pkg_name.replace('\n', '')
             for pkg_name in rqf.readlines() if pkg_name]

        cython_dep = [el for el in install_reqs if 'Cython' in el][0]
        numpy_dep = [el for el in install_reqs if 'numpy' in el][0]

    if 'Cython' not in installed_packages:
        os.system('pip3 install {}'.format(cython_dep))

    from Cython.Build import cythonize

    if 'numpy' not in installed_packages:
        os.system('pip3 install {}'.format(numpy_dep))

    import numpy as np

    IMPROVE_DIR = 'improveai'
    CYTHON_MODULE_DIR = 'cythonized_feature_encoding'

    print('### LISTDIR ###')
    print(os.listdir(os.sep.join(['.', IMPROVE_DIR, CYTHON_MODULE_DIR])))

    cython_feature_encoding_utils_path_str = \
        os.sep.join(
            [IMPROVE_DIR, CYTHON_MODULE_DIR, 'cythonized_feature_encoding_utils.pyx'])

    cython_feature_encoding_utils_ext = \
        Extension(
            '{}.{}.cythonized_feature_encoding_utils'.format(IMPROVE_DIR, CYTHON_MODULE_DIR),
            sources=[cython_feature_encoding_utils_path_str],
            include_dirs=[np.get_include(), os.sep.join(['.', IMPROVE_DIR, CYTHON_MODULE_DIR])])

    cython_feature_encoder_path_str = \
        os.sep.join(
            [IMPROVE_DIR, CYTHON_MODULE_DIR, 'cythonized_feature_encoder.pyx'])

    cython_feature_encoder_ext = \
        Extension(
            '{}.{}.src.cythonized_feature_encoder'.format(IMPROVE_DIR, CYTHON_MODULE_DIR),
            sources=[cython_feature_encoder_path_str],
            include_dirs=[np.get_include(), os.sep.join(['.', IMPROVE_DIR, CYTHON_MODULE_DIR])])

    # setup(name='improve_trainer',
    #       version='0.1',
    #       description='v6 Decision API',
    #       author='Justin Chapweske',
    #       author_email='',
    #       url='https://github.com/improve-ai/trainer/tree/v6',
    #       ext_modules=cythonize(
    #           [cython_feature_encoding_utils_ext, cython_feature_encoder_ext],
    #           language_level="3"),
    #       packages=find_packages(exclude=['*tests*', '*local_test*', '*tools*']))

    setup(name='improveai',
          version='0.1',
          description='v6 Decision API',
          author='Justin Chapweske',
          author_email='',
          url='https://github.com/improve-ai/python-sdk',
          packages=find_packages(exclude=['*tests*']),
          install_requires=["numpy", "setuptools", "wheel", "Cython"] + install_reqs,
          ext_modules=cythonize([cython_feature_encoding_utils_ext, cython_feature_encoder_ext], language_level="3"),
          include_dirs=[np.get_include(), '.'],
          # package_data={'improveai': [pth_str]},
          include_package_data=True)
