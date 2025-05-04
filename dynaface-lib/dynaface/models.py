import hashlib
import logging
import os
import platform
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import rembg  # type: ignore
import requests
import torch
from dynaface.spiga.inference.config import ModelConfig
from dynaface.spiga.inference.framework import SPIGAFramework
from dynaface.util import VERIFY_CERTS
from facenet_pytorch import MTCNN  # type: ignore
from facenet_pytorch.models.mtcnn import ONet, PNet, RNet  # type: ignore
from rembg.sessions.u2net import U2netSession  # type: ignore
from torch import nn
from torch.nn.functional import interpolate  # type: ignore

# Mac M1 issue - hope to remove some day
# RuntimeError: Adaptive pool MPS: input sizes must be divisible by output sizes.
# https://github.com/pytorch/pytorch/issues/96056#issuecomment-1457633408
# https://github.com/pytorch/pytorch/issues/97109
FIX_MPS_ISSUE = True

# Download Constants
MODEL_VERSION = "1"
REDIRECT_URL = "https://data.heatonresearch.com/dynaface/model-loc.json"
FALLBACK_URL = f"https://data.heatonresearch.com/dynaface/model/{MODEL_VERSION}/dynaface_models.zip"
EXPECTED_SHA256 = "c18f9c038b65d7486e7f9e081506bc69cbbc5719680eb31b1bafa8235ca6aa4d"

# Global variables (now explicitly typed as Optional)
_model_path: Optional[str] = None
_device: str = "?"  # Default to CPU
mtcnn_model: Optional[Union[MTCNN, "MTCNN2"]] = None
spiga_model: Optional[SPIGAFramework] = None
rembg_session: Optional[U2netSession] = None

SPIGA_MODEL = "wflw"

logger = logging.getLogger(__name__)


def imresample_mps(img: torch.Tensor, sz: Union[int, Tuple[int, ...]]) -> torch.Tensor:
    # Move the tensor to the CPU and perform interpolation on the CPU before sending it to "mps"
    img_cpu = img.to("cpu")
    im_data = cast(torch.Tensor, interpolate(img_cpu, size=sz, mode="area"))
    return im_data.to("mps")


class MTCNN2(MTCNN):
    def __init__(
        self,
        image_size: int = 160,
        margin: int = 0,
        min_face_size: int = 20,
        thresholds: List[float] = [0.6, 0.7, 0.7],
        factor: float = 0.709,
        post_process: bool = True,
        select_largest: bool = True,
        selection_method: Optional[str] = None,
        keep_all: bool = False,
        device: Optional[str] = None,
        path: str = "",  # now a required string (do not use None)
    ) -> None:
        nn.Module.__init__(self)
        # Don't call the parent constructor, as it initializes MTCNN with default values
        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.selection_method = selection_method

        self.pnet = PNet(pretrained=False)
        self.load_weights(self.pnet, os.path.join(path, "pnet.pt"))
        self.rnet = RNet(pretrained=False)
        self.load_weights(self.rnet, os.path.join(path, "rnet.pt"))
        self.onet = ONet(pretrained=False)
        self.load_weights(self.onet, os.path.join(path, "onet.pt"))

        self.device = torch.device("cpu")
        if device is not None:
            self.device = torch.device(device)
            self.to(device)

        if not self.selection_method:
            self.selection_method = "largest" if self.select_largest else "probability"

    def load_weights(self, net: nn.Module, filename: str) -> None:
        try:
            state_dict = cast(Dict[str, torch.Tensor], torch.load(filename, map_location="cpu"))  # type: ignore
            net.load_state_dict(state_dict)
        except Exception as e:
            raise ValueError(f"Error loading model weights from {filename}: {str(e)}")


def _init_mtcnn() -> None:
    global mtcnn_model
    if _device == "?":
        raise ValueError("Device not initialized. Call init_models() first.")

    if _device == "mps" and FIX_MPS_ISSUE:
        device = "cpu"
    else:
        device = _device

    if _model_path is None:
        mtcnn_model = MTCNN(keep_all=True, device=device)
    else:
        mtcnn_model = MTCNN2(keep_all=True, device=device, path=_model_path)


def _init_spiga() -> None:
    global spiga_model
    config = ModelConfig(dataset_name=SPIGA_MODEL, load_model_url=False)
    config.model_weights_path = _model_path
    spiga_model = SPIGAFramework(config, device=torch.device(_device))


def _init_rembg() -> None:
    global rembg_session
    if _model_path is None:
        raise ValueError("Model path not set. Call init_models() first.")
    os.environ["U2NET_HOME"] = _model_path
    rembg_session = rembg.new_session(model_name="u2net")  # type: ignore


def download_models(
    path: Optional[Union[str, Path]] = None, verify_hash: bool = True
) -> str:
    # Accept either a string or Path; if None, use the default directory.
    if path is None:
        path = Path.home() / ".dynaface" / "models"
    elif isinstance(path, str):
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    model_file = path / "spiga_wflw.pt"  # Check for this file

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
    _init_mtcnn()
    _init_spiga()
    _init_rembg()


def unload_models() -> None:
    global _model_path, _device, mtcnn_model, spiga_model, rembg_session
    _model_path = None
    _device = "cpu"
    mtcnn_model = None
    spiga_model = None
    rembg_session = None
    torch.cuda.empty_cache()


def are_models_init() -> bool:
    return _device != "?"


def detect_device() -> str:
    if platform.system() == "Darwin" and platform.machine() in {"arm64", "x86_64"}:
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def convert_landmarks(landmarks: Dict[str, Any]) -> List[List[Tuple[int, int]]]:
    return [
        [(int(x[0]), int(x[1])) for x in np.array(landmark)]
        for landmark in landmarks["landmarks"]
    ]
