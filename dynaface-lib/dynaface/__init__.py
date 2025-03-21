try:
    from importlib.metadata import version

    __version__ = version("dynaface")
except:
    __version__ = "unknown"

from .models import init_models, are_models_init, detect_device
