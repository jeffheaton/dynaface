import json
import logging
import logging.config
import logging.handlers
import os
import sys

import torch
import version
from dynaface_window import DynafaceWindow
from facial_analysis.facial import STD_PUPIL_DIST, DEFAULT_TILT_THRESHOLD
import facial_analysis
from jth_ui.app_jth import AppJTH, get_library_version
from pillow_heif import register_heif_opener
import jth_ui.utl_settings as utl_settings


logger = logging.getLogger(__name__)

# Constants for settings keys
SETTING_PD = "pd"
SETTING_LOG_LEVEL = "log_level"
SETTING_ACC = "accelerator"
SETTING_TILT_THRESHOLD = "tilt"

register_heif_opener()

# https://stackoverflow.com/questions/75746637/how-to-suppress-qt-pointer-dispatch-warning


class AppDynaface(AppJTH):
    def __init__(self):
        try:
            super().__init__(
                app_name="Dynaface",
                app_author="HeatonResearch",
                copyright="Copyright 2024 by Jeff Heaton, released under the <a href='https://opensource.org/license/mit/'>MIT License</a>",
                version=version.VERSION,
                bundle_id="com.heatonresearch.dynaface",
            )

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
        facial_analysis.facial.AnalyzeFace.pd = utl_settings.get_int(
            self.settings, key=SETTING_PD, default=STD_PUPIL_DIST
        )

        # Set the tilt threshold
        facial_analysis.facial.AnalyzeFace.tilt_threshold = utl_settings.get_int(
            self.settings, key=SETTING_TILT_THRESHOLD, default=DEFAULT_TILT_THRESHOLD
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

        # Use accelerator, if requested
        try:
            facial_analysis.init_models(model_path=self.DATA_DIR, device=self.device)
        except Exception as e:
            logger.error(
                f"Error starting AI models on device {self.device}", exc_info=True
            )
            if self.device != "cpu":
                logger.info("Trying CPU as AI device.")
            self.device = "cpu"
            self.settings[SETTING_ACC] = "cpu"
            facial_analysis.init_models(model_path=self.DATA_DIR, device=self.device)

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
        }
