from Cython.Build import cythonize
import numpy as np
import os
from setuptools import Extension
import pyximport

import improveai.settings as improve_settings

cfe = None

# TODO move this someplace else maybe
if improve_settings.USE_CYTHON_BACKEND:

    rel_pth_prfx = \
        os.sep.join(str(os.path.relpath(__file__)).split(os.sep)[:-2])

    path_str = \
        os.sep.join(
            [improve_settings.IMPROVE_ABS_PATH, 'encoder_cython_utils',
             'cythonized_feature_encoding.pyx'])

    fast_feat_enc_ext = \
        Extension(
            'cythonized_feature_encoding',
            sources=[path_str],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[np.get_include()])

    pyximport.install(
        setup_args={
            'install_requires': ["numpy"],
            'ext_modules': cythonize(
                fast_feat_enc_ext,
                language_level="3"),
            'include_dirs': [np.get_include()]})

    import improveai.encoder_cython_utils.cythonized_feature_encoding as cfe
