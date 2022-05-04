import numpy as np
import os
import pyximport

from improveai.settings import CYTHON_BACKEND_AVAILABLE

if CYTHON_BACKEND_AVAILABLE:

    CYTHON_ENCODING_DIR_PATH = os.sep.join(__file__.split(os.sep)[:-1])
    CYTHON_MODULE_DIR = 'cythonized_feature_encoding'
    IMPROVE_DIR = 'improveai'

    try:
        pyximport.install(setup_args={'include_dirs': [np.get_include(), CYTHON_ENCODING_DIR_PATH]})
        # this might raise a ValueError if numpy version differs from the one which was used to build improveai
        import improveai.cythonized_feature_encoding.cythonized_feature_encoding_utils as cfeu
        import improveai.cythonized_feature_encoding.cythonized_feature_encoder as cfe
    except ValueError as verr:
        # if numpy current numpy version is different than the one which was used to build improveai
        # numpy headers might not match in compiled files
        # to address this rebuilding of *.c files is needed
        files_to_delete = \
            [file_name for file_name in os.listdir(CYTHON_ENCODING_DIR_PATH)
             if file_name.endswith('.c') or file_name.endswith('.so')]
        [os.remove(os.sep.join([CYTHON_ENCODING_DIR_PATH, cython_compiled_file]))
         for cython_compiled_file in files_to_delete]
        pyximport.install(setup_args={'include_dirs': [np.get_include(), CYTHON_ENCODING_DIR_PATH]})
        import improveai.cythonized_feature_encoding.cythonized_feature_encoding_utils as cfeu
        import improveai.cythonized_feature_encoding.cythonized_feature_encoder as cfe


