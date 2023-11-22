from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QEvent
import const_values
from jth_ui.tab_graphic import TabGraphic
from facial_analysis.facial import load_face_image
from facial_analysis.calc import AnalyzeFAI, AnalyzeOralCommissureExcursion, AnalyzeBrows, AnalyzeDentalArea, AnalyzeEyeArea
from facial_analysis.video import VideoToVideo
from PyQt6.QtWidgets import QComboBox, QPushButton, QToolBar, QCheckBox, QSpinBox, QGestureEvent, QPinchGesture


class AnalyzeImageTab(TabGraphic):
    def __init__(self, window, path):
        super().__init__(window)
        self.init_toolbar()
        self._face = load_face_image(path,crop=True)
        self.create_graphic(buffer=self._face.render_img)
        self.update_face()
        self._view.scale(1,1)
        self.grabGesture(Qt.GestureType.PinchGesture)

    def init_toolbar(self):
        self._toolbar = QToolBar()
        self._layout.addWidget(self._toolbar)  # Add the toolbar to the layout first

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

    def action_landmarks(self, state):
        self.update_face()

    def action_measures(self, state):
        self.update_face()

    def action_zoom(self, value):
        z = value/100
        self._view.resetTransform()
        self._view.scale(value/100,value/100)

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
            if scaleFactor>1: 
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
