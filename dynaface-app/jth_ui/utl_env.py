import os
import sys
import platform


def is_sandboxed():
    return "APP_SANDBOX_CONTAINER_ID" in os.environ


def get_system_name():
    system = platform.system().lower()
    if system == "darwin":
        return "osx"
    elif system == "windows":
        return "windows"
    else:
        # This covers Linux and other UNIX-like systems
        return "unix"


def is_pyinstaller_bundle():
    return getattr(sys, "frozen", False)
