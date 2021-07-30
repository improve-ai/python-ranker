from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
import numpy as np
import os
import pip


if __name__ == '__main__':

    rel_pth_prfx = \
        os.sep.join(str(os.path.relpath(__file__)).split(os.sep)[:-1])

    pth_str = \
        '{}{}improveai/choosers/choosers_cython_utils/fast_feat_enc.pyx'\
        .format(
            os.sep.join(str(os.path.relpath(__file__)).split(os.sep)[:-1]),
            '' if not rel_pth_prfx else os.sep)

    fast_feat_enc_ext = \
        Extension(
            'fast_feat_enc',
            sources=[pth_str],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[np.get_include()])

    if float(pip.__version__.split('.')[0]) >= 10:
        parse_requirements = pip._internal.req.parse_requirements
    else:
        parse_requirements = pip.req.parse_requirements

    install_reqs = parse_requirements('./requirements.txt')

    setup(name='improveai',
          version='0.1',
          description='v6 Decision API',
          author='Justin Chapweske',
          author_email='',
          url='https://github.com/improve-ai/python-sdk',
          packages=find_packages(exclude=['*tests*']),
          install_requires=
          ["numpy", "setuptools", "wheel", "Cython"] + install_reqs,
          ext_modules=cythonize(fast_feat_enc_ext, language_level="3"),
          include_dirs=[np.get_include()],
          package_data={'improveai': [pth_str]},
          include_package_data=True)
