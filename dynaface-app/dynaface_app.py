import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["MKL_THREADING_LAYER"] = "GNU"

# Need the above thread setting because of this issue:
# https://github.com/numpy/numpy/issues/654
# See note in spgia.augmentors.utils

import logging
import logging.config
import logging.handlers
import sys

import jth_ui.utl_settings as utl_settings
import torch
import version
from dynaface.facial import DEFAULT_TILT_THRESHOLD, STD_PUPIL_DIST
from dynaface_window import DynafaceWindow
from jth_ui.app_jth import AppJTH, get_library_version
from pillow_heif import register_heif_opener

import dynaface

logger = logging.getLogger(__name__)

# Constants for settings keys
SETTING_PD = "pd"
SETTING_LOG_LEVEL = "log_level"
SETTING_ACC = "accelerator"
SETTING_TILT_THRESHOLD = "tilt"
SETTING_DYNAMIC_ADJUST = "dynamic"
SETTING_SMOOTH = "smooth"

DEFAULT_DYNAMIC_ADJUST = 2
DEFAULT_SMOOTH = 2

register_heif_opener()

# https://stackoverflow.com/questions/75746637/how-to-suppress-qt-pointer-dispatch-warning


class AppDynaface(AppJTH):
    def __init__(self):
        try:
            super().__init__(
                app_name="Dynaface",
                app_author="HeatonResearch",
                copyright="Copyright 2025 by Jeff Heaton, released under the <a href='https://www.apache.org/licenses/LICENSE-2.0'>Apache 2.0 License</a>",
                version=version.VERSION,
                bundle_id="com.heatonresearch.dynaface",
            )
            self.dynamic_adjust = DEFAULT_DYNAMIC_ADJUST
            self.data_smoothing = DEFAULT_SMOOTH
            self.tilt_threshold = DEFAULT_TILT_THRESHOLD

            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.DATA_DIR = os.path.join(self.BASE_DIR, "data")

            self.main_window = DynafaceWindow(app=self, app_name=self.APP_NAME)
            self.main_window.show()

            self.load_dynaface_settings()

            logging.info(f"Using device: {self.device}")
            v = get_library_version("torch")
            logging.info(f"Torch version: {v}")
            v = get_library_version("facenet-pytorch")
            logging.info(f"Facenet-pytorch version: {v}")

        except Exception as e:
            logger.error("Error running app", exc_info=True)

    def load_dynaface_settings(self):
        # Set logging level
        level = utl_settings.get_str(
            self.settings, key=SETTING_LOG_LEVEL, default="INFO"
        )
        logging_level = getattr(logging, level)
        self.change_log_level(logging_level)

        # Set pupillary distance (PD)
        dynaface.facial.AnalyzeFace.pd = utl_settings.get_int(
            self.settings, key=SETTING_PD, default=STD_PUPIL_DIST
        )

        # Set the tilt threshold
        self.tilt_threshold = utl_settings.get_int(
            self.settings, key=SETTING_TILT_THRESHOLD, default=DEFAULT_TILT_THRESHOLD
        )
        # Set the dynamic adjust
        self.dynamic_adjust = utl_settings.get_int(
            self.settings, key=SETTING_DYNAMIC_ADJUST, default=DEFAULT_DYNAMIC_ADJUST
        )

        # Set the data smoothing
        self.data_smoothing = utl_settings.get_int(
            self.settings, key=SETTING_SMOOTH, default=DEFAULT_SMOOTH
        )

        # accelerator device
        acc = utl_settings.get_bool(self.settings, key=SETTING_ACC, default=True)

        if acc:
            # Detect CUDA, MPS, or failing that, CPU
            has_mps = False
            if torch.backends.mps.is_available():
                if torch.backends.mps.is_built():
                    has_mps = True
            self.device = (
                "mps" if has_mps else "gpu" if torch.cuda.is_available() else "cpu"
            )
            logger.info(f"PyTorch Device: {self.device}")
        else:
            self.device = "cpu"

        #

        # Use accelerator, if requested
        try:
            dynaface.models.init_models(model_path=self.DATA_DIR, device=self.device)
        except Exception as e:
            logger.error(
                f"Error starting AI models on device {self.device}", exc_info=True
            )
            if self.device != "cpu":
                logger.info("Trying CPU as AI device.")
            self.device = "cpu"
            self.settings[SETTING_ACC] = "cpu"
            dynaface.models.init_models(model_path=self.DATA_DIR, device=self.device)

    def shutdown(self):
        try:
            super().shutdown()
            sys.exit(0)
        except Exception as e:
            logger.error("Error shutting down app", exc_info=True)

    def init_settings(self):
        self.settings = {
            SETTING_PD: STD_PUPIL_DIST,
            SETTING_LOG_LEVEL: "INFO",
            SETTING_ACC: True,
            SETTING_TILT_THRESHOLD: DEFAULT_TILT_THRESHOLD,
            SETTING_DYNAMIC_ADJUST: DEFAULT_DYNAMIC_ADJUST,
            SETTING_SMOOTH: DEFAULT_SMOOTH,
        }


if __name__ == "__main__":
    app = AppDynaface()
    app.exec()
    app.shutdown()
