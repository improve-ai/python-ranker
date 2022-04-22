import os


def gcc_and_py3_dev_installed():
    gcc_installed = os.system('gcc -v > /dev/null 2>&1') == 0
    py3_dev_installed = os.system('python3-config --help > /dev/null 2>&1') == 0
    return gcc_installed and py3_dev_installed


CYTHON_BACKEND_AVAILABLE = gcc_and_py3_dev_installed()  # False
DEBUG = False

IMPROVE_ABS_PATH = os.sep.join(__file__.split(os.sep)[:-1])
