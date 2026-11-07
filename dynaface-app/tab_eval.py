import logging
import math

import dlg_modal
from dynaface.measures import AnalyzeDentalArea, AnalyzeEyeArea
from dynaface.video import FrameSeries
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QTextEdit, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


def _number(value, places):
    return "n/a" if math.isnan(value) else f"{value:.{places}f}"


def evaluation_html(evaluation):
    """The Evaluation report: the eyes at the widest smile and the mouth at the
    widest eye opening, or a note when no frontal frame has the measures to
    compare (a turned or profile frame never carries them: FrameSeries reports
    a measure as NaN on a frame whose view it doesn't suit)."""
    if evaluation is None:
        return "No frontal frame has the eye and dental measures to compare."

    def heading(title, moment):
        return f"<b>{title}</b> (frame {moment.frame}, {_number(moment.time, 2)} s)"

    def eyes(moment):
        return [
            f"Eye area: left {_number(moment.eye_left, 1)}, "
            f"right {_number(moment.eye_right, 1)}, "
            f"total {_number(moment.eye_total, 1)} mm²",
            f"Left/right: {_number(moment.eye_left_over_right, 3)}, "
            f"right/left: {_number(moment.eye_right_over_left, 3)}",
        ]

    smile, opening = evaluation.widest_smile, evaluation.widest_eyes
    lines = [
        heading("Max dental display", smile),
        f"Dental area: {_number(smile.dental_area, 1)} mm²",
        *eyes(smile),
        "",
        heading("Max eye opening", opening),
        *eyes(opening),
        f"Dental area: {_number(opening.dental_area, 1)} mm²",
    ]
    return "<br>".join(lines)


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
        series = FrameSeries.from_states(
            analyze._frames,
            [AnalyzeEyeArea(), AnalyzeDentalArea()],
            start=analyze._frame_begin,
            stop=analyze._frame_end,
            fps=analyze.frame_rate,
        )
        html = evaluation_html(series.evaluate())
        self._window._background_queue.append(lambda: self.text_edit.setHtml(html))

    def generate(self, analyze):
        logger.info("Evaluate video")
        f = lambda: self.exec_eval(analyze)
        dlg_modal.display_please_wait(window=self, f=f, message="Evaluating video")
