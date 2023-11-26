import const_values
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class AboutTab(QWidget):
    def __init__(self, window):
        super().__init__()
        self._window = window
        text = f"""
<H1>{const_values.APP_NAME} {self._window.app.VERSION}</H1>
{self._window.app.COPYRIGHT}
<br>
<hr>
This program implements the cellular automata described in the paper:<br>
Heaton, J. March (2018). Evolving continuous cellular automata for aesthetic objectives. <i>Genetic Programming Evolvable Machines<i>.<br><a href='https://doi.org/10.1007/s10710-018-9336-1'>https://doi.org/10.1007/s10710-018-9336-1</a>
<hr>
Log path: {self._window.app.LOG_DIR}
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
