import os

def get_hook_dirs():
    print("**HERE2")
    return [os.path.dirname(__file__)]

def get_PyInstaller_tests():
    return [os.path.dirname(__file__)]