from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension


if __name__ == '__main__':

    setup(name='python-sdk',
          version='0.1',
          description='v6 Decision API',
          author='Justin Chapweske',
          author_email='',
          url='https://github.com/improve-ai/python-sdk',
          extensions=cythonize(
                ['utils/cython_experiments/hw.pyx',
                 'choosers/choosers_cython_utils/fast_feat_enc.pyx'],
                language_level="3"))
