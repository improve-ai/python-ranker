import numpy as np
import os
import pyximport
USE_CYTHON_BACKEND = False

CYTHON_ENCODING_DIR_PATH = os.sep.join(__file__.split(os.sep)[:-1])
pyximport.install(setup_args={'include_dirs': [np.get_include(), CYTHON_ENCODING_DIR_PATH]})

import improveai.cythonized_feature_encoding.cythonized_feature_encoding_utils as cfeu
import improveai.cythonized_feature_encoding.cythonized_feature_encoder as cfe
