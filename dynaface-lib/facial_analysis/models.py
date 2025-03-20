import os
import platform
import zipfile
from pathlib import Path

import facenet_pytorch
import numpy as np
import rembg
import requests
import torch
from facenet_pytorch import MTCNN
from facenet_pytorch.models.mtcnn import ONet, PNet, RNet
from facial_analysis.spiga.inference.config import ModelConfig
from facial_analysis.spiga.inference.framework import SPIGAFramework
from torch import nn

# Mac M1 issue - hope to remove some day
# RuntimeError: Adaptive pool MPS: input sizes must be divisible by output sizes.
# https://github.com/pytorch/pytorch/issues/96056#issuecomment-1457633408
# https://github.com/pytorch/pytorch/issues/97109
FIX_MPS_ISSUE = True

# Other values
_model_path = None
_device = None
mtcnn_model = None
spiga_model = None
rembg_session = None

SPIGA_MODEL = "wflw"

from torch.nn.functional import interpolate


def imresample_mps(img, sz):
    # Move the tensor to the CPU
    img_cpu = img.to("cpu")
    # Perform the interpolation on the CPU
    im_data = interpolate(img_cpu, size=sz, mode="area")
    return im_data.to("mps")


class MTCNN2(MTCNN):
    def __init__(
        self,
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        select_largest=True,
        selection_method=None,
        keep_all=False,
        device=None,
        path=None,
    ):
        nn.Module.__init__(self)

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
            self.device = device
            self.to(device)

        if not self.selection_method:
            self.selection_method = "largest" if self.select_largest else "probability"

    def load_weights(self, net, filename):
        state_dict = torch.load(filename)
        net.load_state_dict(state_dict)


def _init_mtcnn() -> None:
    global mtcnn_model

    if _device == "mps" and FIX_MPS_ISSUE:
        device = "cpu"
    else:
        device = _device
        # print(facenet_pytorch.models.utils.imresample)
        # facenet_pytorch.models.utils.detect_face.imresample = imresample_mps
        # print(facenet_pytorch.models.utils.detect_face.imresample)

    if _model_path is None:
        mtcnn_model = MTCNN(keep_all=True, device=device)
    else:
        mtcnn_model = MTCNN2(keep_all=True, device=device, path=_model_path)


def _init_spiga() -> None:
    global spiga_model

    config = ModelConfig(dataset_name=SPIGA_MODEL, load_model_url=False)
    config.model_weights_path = _model_path
    spiga_model = SPIGAFramework(config, device=_device)


def _init_rembg() -> None:
    global rembg_session
    os.environ["U2NET_HOME"] = _model_path
    rembg_session = rembg.new_session(model_name="u2net")


def download_models(path: str = None) -> str:
    # Set default path if none provided
    if path is None:
        path = Path.home() / ".dynaface" / "models"
    else:
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    model_file = path / "spiga_wflw.pt"  # Check for this file

    if model_file.exists():
        return str(path)

    zip_url = "https://data.heatonresearch.com/dynaface/model/1/dynaface_models.zip"
    zip_path = path / "dynaface_models.zip"

    # Download the ZIP file
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path)

    # Remove the ZIP file after extraction
    zip_path.unlink()

    return str(path)


def init_models(model_path: str, device: str) -> None:
    global _model_path, _device

    _model_path = model_path
    _device = device

    _init_mtcnn()
    _init_spiga()
    _init_rembg()


def are_models_init() -> bool:
    global _device
    return _device is not None


def detect_device() -> str:
    if platform.system() == "Darwin" and platform.machine() in {"arm64", "x86_64"}:
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def convert_landmarks(landmarks):
    return [
        [(int(x[0]), int(x[1])) for x in np.array(landmark)]
        for landmark in landmarks["landmarks"]
    ]
