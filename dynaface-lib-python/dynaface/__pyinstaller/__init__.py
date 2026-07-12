import os
from typing import List


def get_hook_dirs() -> List[str]:
    return [os.path.dirname(__file__)]


def get_PyInstaller_tests() -> List[str]:
    return [os.path.dirname(__file__)]
