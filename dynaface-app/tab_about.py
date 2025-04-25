from jth_ui.app_jth import get_library_version
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from version import BUILD_DATE

import dynaface
from jth_ui import app_const, utl_log


class AboutTab(QWidget):
    def __init__(self, window):
        super().__init__()
        self._window = window
        device = dynaface.models.detect_device()
        current_device = dynaface.models._device
        v1 = get_library_version("torch")
        v2 = get_library_version("facenet-pytorch")
        text = f"""
<H1>{app_const.APP_NAME} {app_const.VERSION}</H1>
{app_const.COPYRIGHT}
<br>
Produced in collaboration with [insert names], [Johns Hopkins reference]. 
<br>
This program is for education and research purposes only.
<hr>
This program implements the algorithms described in the paper:<br>
[insert actual paper cite]
<hr>
Build Date: {BUILD_DATE} <br>
Log path: {utl_log.get_log_dir()} <br>
Torch version: {v1}<br>
Dynaface Library Version: {dynaface.__version__}<br>
Facenet-pytorch version: {v2}<br>
Processor in use: {current_device} (detected: {device})<br>
Data Path: {self._window.app.DATA_DIR} <br>
"""

        # Create the QLabel with the hyperlink
        self.label = QLabel(text, self)
        self.label.setOpenExternalLinks(True)

        # Set a layout for the CustomTab and add the label to it
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        # Align the content to the top
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def on_close(self):
        pass

    def on_resize(self):
        pass
