from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
import facial_analysis


class AboutTab(QWidget):
    def __init__(self, window):
        super().__init__()
        self._window = window
        device = facial_analysis.detect_device()
        current_device = facial_analysis.models._device
        text = f"""
<H1>{self._window.app.APP_NAME} {self._window.app.VERSION}</H1>
{self._window.app.COPYRIGHT}
<br>
Produced in collaboration with [insert names], [Johns Hopkins reference]. 
<br>
This program is for education and research purposes only.
<hr>
This program implements the algorithms described in the paper:<br>
[insert actual paper cite]
<hr>
Log path: {self._window.app.LOG_DIR} <br>
Processor in use: {current_device} (detected: {device})
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
