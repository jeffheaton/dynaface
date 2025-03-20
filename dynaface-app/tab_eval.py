import logging

import dlg_modal
from dynaface.facial import AnalyzeFace
from dynaface.measures import AnalyzeDentalArea, AnalyzeEyeArea, all_measures
from jth_ui.app_jth import get_library_version
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QTextEdit, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class TabEval(QWidget):
    def __init__(self, window):
        super().__init__()
        self._window = window

        # Create the QTextEdit for displaying text
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        # Create a button for copying text
        self.copy_button = QPushButton("Copy All Text", self)
        self.copy_button.clicked.connect(self.copy_all_text)

        # Set a layout for the tab and add the text edit and button to it
        layout = QVBoxLayout(self)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.copy_button)

        # Align the content to the top
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def copy_all_text(self):
        self.text_edit.selectAll()
        self.text_edit.copy()

    def on_close(self):
        pass

    def on_resize(self):
        pass

    def exec_eval(self, analyze):
        import dynaface_app

        measures = [AnalyzeEyeArea(), AnalyzeDentalArea()]
        tilt_threshold = app.tilt_threshold
        face = AnalyzeFace(measures, tilt_threshold=tilt_threshold)
        frames = range(analyze._frame_begin, analyze._frame_end, 1)

        max_eye_idx = max_eye = -1
        max_eye_data = ""
        max_smile_idx = max_smile = -1
        max_smile_data = ""

        for i in frames:
            frame = analyze._frames[i]
            face.load_state(frame)
            rec = face.analyze()
            el = rec["eye.l"]
            er = rec["eye.r"]
            ea = el + er
            da = rec["dental_area"]

            if max_eye == -1 or ea > max_eye:
                # max_eye_idx = i
                max_eye = ea
                max_eye_data = f"Max Dental (frame:{i}: Dental area: {round(da,1)})"

            if max_smile == -1 or da > max_smile:
                # max_smile_idx = i
                max_smile = da

                ratio_lr = round(el / er, 3)
                ratio_rl = round(er / el, 3)
                max_smile_data = f"Max ocular (frame:{i}): left={round(el,1)}, right={round(er,1)}, lr={ratio_lr}, rl={ratio_rl}, total={round(ea,1)}"

        self._window._background_queue.append(
            lambda: self.text_edit.setHtml(f"{max_eye_data}<br>{max_smile_data}")
        )

    def generate(self, analyze):
        logger.info("Evaluate video")
        f = lambda: self.exec_eval(analyze)
        dlg_modal.display_please_wait(window=self, f=f, message="Evaluating video")
