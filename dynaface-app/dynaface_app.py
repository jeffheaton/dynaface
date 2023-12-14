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
from jth_ui.app_jth import AppJTH
from pillow_heif import register_heif_opener

logger = logging.getLogger(__name__)

# Constants for settings keys
SETTING_PD = "pd"
SETTING_LOG_LEVEL = "log_level"

register_heif_opener()


class AppDynaface(AppJTH):
    def __init__(self):
        try:
            super().__init__(
                app_name="Dynaface",
                app_author="HeatonResearch",
                copyright="Copyright 2023 by Jeff Heaton, released under the <a href='https://opensource.org/license/mit/'>MIT License</a>",
                version="1.0.0",
                bundle_id="com.heatonresearch.dynaface",
            )

            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.DATA_DIR = os.path.join(self.BASE_DIR, "data")

            self.main_window = DynafaceWindow(app=self, app_name=self.APP_NAME)
            self.main_window.show()

            has_mps = torch.backends.mps.is_built()
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
        self.settings = {SETTING_PD: STD_PUPIL_DIST, SETTING_LOG_LEVEL: "INFO"}
