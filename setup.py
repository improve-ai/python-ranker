from Cython.Build import cythonize
# from distutils.core import setup
from setuptools import setup, Extension
import sys
# from distutils.extension import Extension
import numpy as np
import os


if __name__ == '__main__':

    setup(name='python-sdk',
          version='0.1',
          description='v6 Decision API',
          author='Justin Chapweske',
          author_email='',
          url='https://github.com/improve-ai/python-sdk'  # ,
          # ext_modules=cythonize([
          #     Extension(
          #         "fast_enc",
          #         [pth_str],
          #         include_dirs=[np.get_include()],
          #         define_macros=[
          #                   ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]),
          # install_requires=["numpy"]
          )
