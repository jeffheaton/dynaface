import facial_analysis
from jth_ui.app_jth import get_library_version
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from version import BUILD_DATE


class AnalyzeLaterialTab(QWidget):
    def __init__(self, window, path):
        super().__init__(window)
        self.unsaved_changes = False

        self._auto_update = False

        # Load the face

    def on_close(self):
        pass

    def on_resize(self):
        pass
