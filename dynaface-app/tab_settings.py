import logging

import dynaface_app
from facial_analysis.facial import STD_PUPIL_DIST
import facial_analysis
from PyQt6.QtGui import QIntValidator
from jth_ui import utl_settings
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class SettingsTab(QWidget):
    def __init__(self, window):
        super().__init__()
        self._window = window

        # Create widgets
        lbl_pd = QLabel("Pupillary Distance (PD, mm):", self)
        self._text_pd = QLineEdit(self)
        self._text_pd.setValidator(QIntValidator())
        log_level_label = QLabel("Log Level:", self)
        self._log_combo_box = QComboBox()
        self._log_combo_box.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

        # Create button layout
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.action_save)
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(lambda: self.action_cancel())
        button_layout = QHBoxLayout()
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)

        # Form layout for the options
        form_layout = QFormLayout()
        form_layout.addRow(lbl_pd, self._text_pd)
        form_layout.addRow(log_level_label, self._log_combo_box)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        window.add_tab(self, "Settings")

        settings = self._window.app.settings
        print(settings)
        self._text_pd.setText(
            settings.get(dynaface_app.SETTING_PD, str(STD_PUPIL_DIST))
        )
        utl_settings.set_combo(
            self._log_combo_box,
            settings.get(dynaface_app.SETTING_LOG_LEVEL, "INFO"),
        )

    def on_close(self):
        pass
        # self._window.close_simulator_tabs()

    def action_save(self):
        self.save_values()
        self._window.close_current_tab()

    def action_cancel(self):
        self._window.close_current_tab()

    def on_resize(self):
        pass

    def save_values(self):
        settings = self._window.app.settings

        settings[dynaface_app.SETTING_PD] = self._text_pd.text()
        settings[dynaface_app.SETTING_LOG_LEVEL] = self._log_combo_box.currentText()
        level = settings[dynaface_app.SETTING_LOG_LEVEL]
        logging_level = getattr(logging, level)
        self._window.app.change_log_level(logging_level)
        pd = settings.get(dynaface_app.SETTING_PD, STD_PUPIL_DIST)
        facial_analysis.facial.AnalyzeFace.pd = int(pd)
        self._window.app.save_settings()
