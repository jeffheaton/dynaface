import logging
import os
import sys

import utl_gfx
from facial_analysis.facial import load_face_image
from jth_ui.tab_graphic import TabGraphic
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGestureEvent,
    QHBoxLayout,
    QMainWindow,
    QPinchGesture,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class AnalyzeImageTab(TabGraphic):
    def __init__(self, window, path):
        super().__init__(window)
        tab_layout = QVBoxLayout(self)
        self.init_horizontal_toolbar(tab_layout)

        # Create a horizontal layout for the content of the tab
        content_layout = QHBoxLayout()
        tab_layout.addLayout(content_layout)
        self.init_vertical_toolbar(content_layout)
        self.init_graphics(content_layout)

        # Load the face
        self._face = load_face_image(path, crop=True)
        self.create_graphic(buffer=self._face.render_img)
        self.update_face()
        self._view.scale(1, 1)
        self.grabGesture(Qt.GestureType.PinchGesture)

    def init_horizontal_toolbar(self, layout):
        self._toolbar = QToolBar()
        layout.addWidget(self._toolbar)  # Add the toolbar to the layout first

        # Start Button
        self._btn_start = QPushButton("Reset")
        self._btn_start.clicked.connect(self.start_game)
        self._toolbar.addWidget(self._btn_start)

        self._chk_landmarks = QCheckBox("Landmarks")
        self._toolbar.addWidget(self._chk_landmarks)
        self._chk_landmarks.stateChanged.connect(self.action_landmarks)

        self._chk_measures = QCheckBox("Measures")
        self._toolbar.addWidget(self._chk_measures)
        self._chk_measures.stateChanged.connect(self.action_measures)

        self._spin_zoom = QSpinBox()
        self._toolbar.addWidget(self._spin_zoom)
        self._spin_zoom.setMinimum(1)
        self._spin_zoom.setMaximum(200)
        self._spin_zoom.setSingleStep(5)
        self._spin_zoom.setValue(100)  # Starting value
        self._spin_zoom.setFixedWidth(60)  # Adjust the width as needed
        self._spin_zoom.valueChanged.connect(self.action_zoom)

    def init_vertical_toolbar(self, layout):
        # Add a vertical toolbar (left side of the tab)
        self.left_toolbar = QToolBar("Left Toolbar", self)
        self.left_toolbar.setOrientation(Qt.Orientation.Vertical)
        layout.addWidget(self.left_toolbar)

        # Create a horizontal layout for "All" and "None" buttons
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(0)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Add "All" and "None" buttons to the horizontal layout
        self.all_button = QPushButton("All", self)
        self.none_button = QPushButton("None", self)
        button_width = 50
        self.all_button.setFixedWidth(button_width)
        self.none_button.setFixedWidth(button_width)
        self.buttons_layout.addWidget(self.all_button)
        self.buttons_layout.addWidget(self.none_button)

        # Add the buttons layout to the left toolbar as a widget
        self.buttons_widget = QWidget()
        self.buttons_widget.setLayout(self.buttons_layout)
        self.left_toolbar.addWidget(self.buttons_widget)

        # Create a scrollable area for checkboxes
        self.scroll_area = QScrollArea(self)
        self.scroll_area_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget)
        self.left_toolbar.addWidget(self.scroll_area)

        # Store checkboxes in a list for easy access
        self.checkboxes = []

        # Add check boxes to the scrollable area
        for i in range(10):
            checkbox = QCheckBox(f"Option {chr(65+i)}")
            self.scroll_area_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        # Connect buttons to slot functions
        self.all_button.clicked.connect(self.check_all)
        self.none_button.clicked.connect(self.uncheck_all)

    def action_landmarks(self, state):
        self.update_face()

    def action_measures(self, state):
        self.update_face()

    def action_zoom(self, value):
        z = value / 100
        self._view.resetTransform()
        self._view.scale(value / 100, value / 100)

    def update_face(self):
        self._face.render_reset()
        if self._chk_measures.isChecked():
            self._face.analyze()
        if self._chk_landmarks.isChecked():
            self._face.draw_landmarks(numbers=True)
        self.update_graphic(resize=False)

    def gestureEvent(self, event: QGestureEvent):
        pinch = event.gesture(Qt.GestureType.PinchGesture)
        if isinstance(pinch, QPinchGesture):
            scaleFactor = pinch.scaleFactor()
            if scaleFactor > 1:
                new_value = self._spin_zoom.value() + 2
            else:
                new_value = self._spin_zoom.value() - 2
            self._spin_zoom.setValue(new_value)
            return True
        return super().event(event)

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            return self.gestureEvent(event)
        return super().event(event)

    def on_close(self):
        pass

    def on_resize(self):
        pass

    def on_copy(self):
        logging.info(f"Copy image: {self._face.render_img.shape}")
        utl_gfx.copy_image_to_clipboard(self._face.render_img)

    def check_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def uncheck_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
