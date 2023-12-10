import sys
import io
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap
import plotly.graph_objects as go
from PIL import Image

# Create a sample Plotly graph
fig = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])])

# Render the graph to an in-memory image
img_buffer = io.BytesIO()
fig.write_image(img_buffer, format='png')
img_buffer.seek(0)  # Move to the beginning of the buffer

# Create a PyQt6 application
app = QApplication(sys.argv)
window = QMainWindow()
central_widget = QWidget()
layout = QVBoxLayout(central_widget)

# Create a QLabel to display the image
label = QLabel()
pixmap = QPixmap()
pixmap.loadFromData(img_buffer.getvalue())
label.setPixmap(pixmap)

# Add the label to the window
layout.addWidget(label)
window.setCentralWidget(central_widget)

# Show the window
window.show()

# Start the application's event loop
sys.exit(app.exec())