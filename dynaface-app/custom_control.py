from PyQt6.QtWidgets import QCheckBox


class CheckingCheckBox(QCheckBox):
    def __init__(self, title, parent):
        super().__init__(title)
        self._parent = parent

    def mousePressEvent(self, event):
        # Add your logic here to decide whether to change state
        if self._parent.checkCheckboxEvent(self):  # Replace with your condition
            super().mousePressEvent(event)  # Allow the state change
        else:
            event.ignore()  # Ignore the event to prevent state change
