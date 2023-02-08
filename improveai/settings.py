import os


def gcc_and_py3_dev_installed() -> bool:
    """
    Checks if gcc and python3 headers are installed.

    Returns
    -------
    bool
        True if both gcc and python3 headers are installed otherwise False

    """
    gcc_installed = os.system('gcc -v > /dev/null 2>&1') == 0
    py3_dev_installed = os.system('python3-config --help > /dev/null 2>&1') == 0
    return gcc_installed and py3_dev_installed

# # TODO use cython once debugged
# CYTHON_BACKEND_AVAILABLE = False
# docs for module level variables like CYTHON_BACKEND_AVAILABLE are placed below them
CYTHON_BACKEND_AVAILABLE = gcc_and_py3_dev_installed()
"""
Indicates if cython backend is usable with python-SDK
"""


DEBUG = False
"""
Indicates if debug level messages should be shown
"""

IMPROVE_ABS_PATH = os.sep.join(__file__.split(os.sep)[:-1])
MAX_TRACK_THREADS = 16
