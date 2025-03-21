import os

try:
    from importlib.metadata import version

    __version__ = version(__name__)
except:
    __version__ = "unknown"


def get_hook_dirs():
    return [os.path.dirname(__file__)]


def get_PyInstaller_tests():
    return [os.path.dirname(__file__)]
