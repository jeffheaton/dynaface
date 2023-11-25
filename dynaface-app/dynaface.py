import json
import logging
import logging.config
import logging.handlers
import os
import plistlib
import sys
import webbrowser

import torch
from facial_analysis.find_face import FindFace
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QMenu, QMenuBar, QTabWidget

import const_values
import tab_settings
import tab_splash
from jth_ui.app_jth import AppJTH
from jth_ui.window_jth import MainWindowJTH
from tab_about import AboutTab
from tab_analyze_image import AnalyzeImageTab
from dynaface_app import AppDynaface

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    app = AppDynaface()
    app.exec()
    app.shutdown()
