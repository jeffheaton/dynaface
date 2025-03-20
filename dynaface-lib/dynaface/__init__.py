from importlib.metadata import version, PackageNotFoundError
import tomllib
from pathlib import Path


def get_version():
    try:
        return version("dynaface")  # Works if installed
    except PackageNotFoundError:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with pyproject_path.open("rb") as f:
                data = tomllib.load(f)
            return data["tool"]["poetry"]["version"]
        return "unknown"  # Default if both methods fail


__version__ = get_version()

from .models import init_models, are_models_init, detect_device
