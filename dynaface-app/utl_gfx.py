import io
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QClipboard, QPixmap
import plotly.graph_objects as go
from PIL import Image


def opencv_img_to_qimage(opencv_img):
    # Convert from BGR to RGB
    # rgb_image = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

    # Create a QImage from the RGB image
    h, w, ch = opencv_img.shape
    bytes_per_line = ch * w
    return QImage(opencv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


def copy_image_to_clipboard(opencv_img):
    # Check if a QApplication already exists, if not, create one
    app = QApplication.instance()
    if not app:  # If it does not exist, create a QApplication
        app = QApplication(sys.argv)

    # Convert OpenCV image to QImage
    image = opencv_img_to_qimage(opencv_img)

    # Copy image to clipboard
    clipboard = QApplication.clipboard()
    clipboard.setImage(image, mode=QClipboard.Mode.Clipboard)

def poltly_to_pixmap(fig):
    img_buffer = io.BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)  # Move to the beginning of the buffer
    pixmap = QPixmap()
    pixmap.loadFromData(img_buffer.getvalue())
    return pixmap

