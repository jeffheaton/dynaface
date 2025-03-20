import logging

from dynaface.facial import DEFAULT_TILT_THRESHOLD, STD_PUPIL_DIST
from jth_ui import utl_settings
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import dynaface

logger = logging.getLogger(__name__)


class SettingsTab(QWidget):
    def __init__(self, window):
        super().__init__()
        app = QApplication.instance()

        self._window = window

        # Create widgets
        lbl_pd = QLabel("Pupillary Distance (PD, mm):", self)
        self._text_pd = QLineEdit(self)
        self._text_pd.setValidator(QIntValidator())

        lbl_tilt = QLabel("Correct tilt greater than (deg, -1 to disable):", self)
        self._text_tilt = QLineEdit(self)
        self._text_tilt.setValidator(QIntValidator())

        lbl_dynamic_adjust = QLabel("Crop/zoom/tilt smoothing (1 to disable):", self)
        self._text_dynamic_adjust = QLineEdit(self)
        self._text_dynamic_adjust.setValidator(QIntValidator())

        lbl_data_smooth = QLabel("Data smoothing (1 to disable):", self)
        self._text_data_smooth = QLineEdit(self)
        self._text_dynamic_adjust.setValidator(QIntValidator())

        log_level_label = QLabel("Log Level:", self)
        self._log_combo_box = QComboBox()
        self._log_combo_box.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        lbl_acc = QLabel("Use Accelerator (GPU)")
        self._chk_accelerator = QCheckBox()

        # Create button layout
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.action_save)
        reset_button = QPushButton("Reset", self)
        reset_button.clicked.connect(self.action_reset)
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(lambda: self.action_cancel())
        button_layout = QHBoxLayout()
        button_layout.addWidget(save_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(cancel_button)

        # Form layout for the options
        form_layout = QFormLayout()
        form_layout.addRow(lbl_pd, self._text_pd)
        form_layout.addRow(lbl_tilt, self._text_tilt)
        form_layout.addRow(lbl_dynamic_adjust, self._text_dynamic_adjust)
        form_layout.addRow(lbl_data_smooth, self._text_data_smooth)
        form_layout.addRow(log_level_label, self._log_combo_box)
        form_layout.addRow(lbl_acc, self._chk_accelerator)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        window.add_tab(self, "Settings")

        settings = self._window.app.settings
        self._text_pd.setText(str(dynaface.facial.AnalyzeFace.pd))
        tilt_threshold = app.tilt_threshold
        self._text_tilt.setText(str(tilt_threshold))

        self._text_dynamic_adjust.setText(str(self._window.app.dynamic_adjust))
        self._text_data_smooth.setText(str(self._window.app.data_smoothing))

        utl_settings.set_combo(
            self._log_combo_box,
            settings.get(dynaface_app.SETTING_LOG_LEVEL, "INFO"),
        )
        self._chk_accelerator.setChecked(settings.get(dynaface_app.SETTING_ACC, True))

    def on_close(self):
        pass
        # self._window.close_simulator_tabs()

    def action_save(self):
        self.save_values()
        self._window.close_current_tab()

    def action_reset(self):
        import dynaface_app

        self._text_pd.setText(str(dynaface.facial.STD_PUPIL_DIST))
        self._text_tilt.setText(str(dynaface.facial.DEFAULT_TILT_THRESHOLD))
        self._text_dynamic_adjust.setText(str(dynaface_app.DEFAULT_DYNAMIC_ADJUST))
        self._text_data_smooth.setText(str(dynaface_app.DEFAULT_SMOOTH))
        utl_settings.set_combo(self._log_combo_box, "INFO")

    def action_cancel(self):
        self._window.close_current_tab()

    def on_resize(self):
        pass

    def save_values(self):
        import dynaface_app

        settings = self._window.app.settings

        settings[dynaface_app.SETTING_PD] = utl_settings.parse_int(
            self._text_pd.text(), default=STD_PUPIL_DIST
        )
        settings[dynaface_app.SETTING_LOG_LEVEL] = self._log_combo_box.currentText()
        settings[dynaface_app.SETTING_ACC] = self._chk_accelerator.isChecked()
        settings[dynaface_app.SETTING_TILT_THRESHOLD] = utl_settings.parse_int(
            self._text_tilt.text(), default=DEFAULT_TILT_THRESHOLD
        )

        dynamic_adjust = utl_settings.parse_int(
            self._text_dynamic_adjust.text(),
            default=dynaface_app.DEFAULT_DYNAMIC_ADJUST,
        )

        data_smoothing = utl_settings.parse_int(
            self._text_data_smooth.text(), default=dynaface_app.DEFAULT_SMOOTH
        )

        if dynamic_adjust < 1:
            dynamic_adjust = 1
        if data_smoothing < 1:
            data_smoothing = 1

        settings[dynaface_app.SETTING_DYNAMIC_ADJUST] = dynamic_adjust
        settings[dynaface_app.SETTING_SMOOTH] = data_smoothing

        level = settings[dynaface_app.SETTING_LOG_LEVEL]
        logging_level = getattr(logging, level)
        self._window.app.change_log_level(logging_level)
        self._window.app.save_settings()
        self._window.app.load_dynaface_settings()

        # if dynaface.models._device != settings[dynaface_app.SETTING_ACC]:
        #    self._window.display_message_box(
        #        "Changes to accelerator settings will be effective on application restart."
        #    )
