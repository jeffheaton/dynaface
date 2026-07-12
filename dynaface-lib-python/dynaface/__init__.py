try:
    from importlib.metadata import version

    __version__ = version("dynaface")
except:
    __version__ = "unknown"
