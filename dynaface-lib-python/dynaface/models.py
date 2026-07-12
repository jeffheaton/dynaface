import gc
import hashlib
import logging
import platform
import zipfile
from pathlib import Path
from typing import Optional, Union

import onnxruntime as ort
import requests

from dynaface.dynaface_onnx import DynafaceOnnxInference
from dynaface.util import VERIFY_CERTS

# Download Constants
MODEL_VERSION = "2"
REDIRECT_URL = "https://data.heatonresearch.com/dynaface/model-loc.json"
FALLBACK_URL = f"https://data.heatonresearch.com/dynaface/model/{MODEL_VERSION}/dynaface_models.zip"
EXPECTED_SHA256 = "8adc1246c00e8ea5a53e81e8000717b4541eabba59639ae178a07a799b5c57de"

# Global variables (now explicitly typed as Optional)
_model_path: Optional[str] = None
_device: str = "?"  # Default to CPU
onnx_model: Optional[DynafaceOnnxInference] = None

logger = logging.getLogger(__name__)


def _init_onnx() -> None:
    global onnx_model
    if _model_path is None:
        raise ValueError("Model path not set. Call init_models() first.")
    # A previous DynafaceOnnxInference's ONNX Runtime sessions (in particular
    # CoreML's) must be fully released before new ones are created: leaving
    # two live session sets in the same process at once has been observed to
    # silently change inference output numerically, even though each session
    # is deterministic on its own. Drop the reference and force collection
    # rather than relying on whenever the GC would otherwise run.
    onnx_model = None
    gc.collect()
    onnx_model = DynafaceOnnxInference(model_dir=_model_path, device=_device)


def download_models(
    path: Optional[Union[str, Path]] = None, verify_hash: bool = True
) -> str:
    # Accept either a string or Path; if None, use the default directory.
    if path is None:
        path = Path.home() / ".dynaface" / "models"
    elif isinstance(path, str):
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    model_file = path / "spiga_wflw.onnx"  # Check for this file

    if model_file.exists():
        return str(path)

    zip_path = path / "dynaface_models.zip"

    # Try to fetch redirected URL for the ZIP file
    try:
        response = requests.get(REDIRECT_URL, timeout=10, verify=VERIFY_CERTS)
        response.raise_for_status()
        model_info = response.json()
        zip_url = model_info[MODEL_VERSION]["url"]
    except Exception:
        zip_url = FALLBACK_URL  # Fallback if redirect fails

    # Try to download ZIP using primary URL; if it fails, try the fallback.
    try:
        logger.info(f"Downloading DynaFace model files from {zip_url}...")
        response = requests.get(zip_url, stream=True, timeout=30, verify=VERIFY_CERTS)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as primary_exception:
        if zip_url == FALLBACK_URL:
            raise primary_exception
        try:
            response = requests.get(
                FALLBACK_URL, stream=True, timeout=30, verify=VERIFY_CERTS
            )
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception:
            raise primary_exception

    # Verify SHA-256 checksum if requested
    if verify_hash:
        sha256 = hashlib.sha256()
        with open(zip_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        if file_hash != EXPECTED_SHA256:
            zip_path.unlink()  # Clean up the downloaded file
            raise ValueError(
                f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {file_hash}. "
                "Set verify_hash=False to skip this check (not recommended)."
            )

    # Extract the ZIP file and remove it afterward
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path)
    zip_path.unlink()

    return str(path)


def init_models(model_path: str, device: str) -> None:
    global _model_path, _device
    _model_path = model_path
    _device = device
    _init_onnx()


def unload_models() -> None:
    global _model_path, _device, onnx_model
    _model_path = None
    _device = "cpu"
    onnx_model = None


def are_models_init() -> bool:
    return _device != "?"


def detect_device() -> str:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return "cuda"
    if platform.system() == "Darwin" and "CoreMLExecutionProvider" in available:
        return "mps"
    return "cpu"
