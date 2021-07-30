from setuptools import Extension, find_packages, setup

import os
import pydoc


if __name__ == '__main__':

    installed_packages = [pkg.name for pkg in pydoc.pkgutil.iter_modules()]

    with open('requirements.txt') as rqf:
        # install_reqs = \
        #     [pkg_name.split('==')[0].replace('\n', '')
        #      for pkg_name in rqf.readlines() if pkg_name]

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
