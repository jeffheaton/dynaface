import json
import logging
import logging.config
import logging.handlers
import os
import sys

import torch
from dynaface_window import DynafaceWindow
from facial_analysis.facial import STD_PUPIL_DIST
import facial_analysis
from jth_ui.app_jth import AppJTH, get_library_version
from pillow_heif import register_heif_opener


logger = logging.getLogger(__name__)

# Constants for settings keys
SETTING_PD = "pd"
SETTING_LOG_LEVEL = "log_level"
SETTING_ACC = "accelerator"

register_heif_opener()

# https://stackoverflow.com/questions/75746637/how-to-suppress-qt-pointer-dispatch-warning


class AppDynaface(AppJTH):
    def __init__(self):
        try:
            super().__init__(
                app_name="Dynaface",
                app_author="HeatonResearch",
                copyright="Copyright 2024 by Jeff Heaton, released under the <a href='https://opensource.org/license/mit/'>MIT License</a>",
                version="1.1.1",
                bundle_id="com.heatonresearch.dynaface",
            )

            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.DATA_DIR = os.path.join(self.BASE_DIR, "data")

            self.main_window = DynafaceWindow(app=self, app_name=self.APP_NAME)
            self.main_window.show()

            has_mps = False
            if torch.backends.mps.is_available():
                if torch.backends.mps.is_built():
                    has_mps = True
            device = "mps" if has_mps else "gpu" if torch.cuda.is_available() else "cpu"
            logger.info(f"PyTorch Device: {device}")

            # Set logging level
            level = self.settings.get(SETTING_LOG_LEVEL, "INFO")
            logging_level = getattr(logging, level)
            self.change_log_level(logging_level)

            try:
                pd = self.settings.get(SETTING_PD, STD_PUPIL_DIST)
                facial_analysis.facial.AnalyzeFace.pd = int(pd)
            except:
                facial_analysis.facial.AnalyzeFace.pd = STD_PUPIL_DIST

            if not self.settings.get(SETTING_ACC, True):
                device = "cpu"

            logging.info(f"Using device: {device}")
            v = get_library_version("torch")
            logging.info(f"Torch version: {v}")
            v = get_library_version("facenet-pytorch")
            logging.info(f"Facenet-pytorch version: {v}")

            try:
                facial_analysis.init_models(model_path=self.DATA_DIR, device=device)
            except Exception as e:
                logger.error(
                    f"Error starting AI models on device {device}", exc_info=True
                )
                if device != "cpu":
                    logger.info("Trying CPU as AI device.")
                device = "cpu"
                self.settings[SETTING_ACC] = "cpu"
                facial_analysis.init_models(model_path=self.DATA_DIR, device=device)
        except Exception as e:
            logger.error("Error running app", exc_info=True)

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
        }
