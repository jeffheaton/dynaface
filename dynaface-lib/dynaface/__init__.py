try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("dynaface")
except PackageNotFoundError:
    __version__ = "unknown"
