import sys

from PyQt6.QtGui import QClipboard, QImage, QPixmap
from PyQt6.QtWidgets import QApplication
import dynaface


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


from PyQt6.QtCore import QRect
from PyQt6.QtGui import QColor, QPixmap


def crop_pixmap(pixmap: QPixmap, pad_px: int) -> QPixmap:
    """
    Crops a pixmap by removing margins with a uniform background color and adding padding.

    :param pixmap: The QPixmap to crop.
    :param pad_px: The number of pixels to pad around the cropped area.
    :return: A cropped and padded QPixmap.
    """
    if pixmap.isNull():
        return pixmap

    # Get the background color from the top-left pixel
    background_color = QColor(pixmap.toImage().pixelColor(0, 0))

    # Find the tightest rectangle that excludes the background
    left, top, right, bottom = pixmap.width(), pixmap.height(), 0, 0
    for x in range(pixmap.width()):
        for y in range(pixmap.height()):
            if pixmap.toImage().pixelColor(x, y) != background_color:
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)

    # If the entire pixmap is the background color, return it as is
    if left > right or top > bottom:
        return pixmap

    # Adjust the rectangle for padding
    left = max(0, left - pad_px)
    top = max(0, top - pad_px)
    right = min(pixmap.width() - 1, right + pad_px)
    bottom = min(pixmap.height() - 1, bottom + pad_px)

    # Crop and return the pixmap
    rect = QRect(left, top, right - left + 1, bottom - top + 1)
    return pixmap.copy(rect)


def load_face_image(
    filename,
    crop=True,
    stats=None,
    tilt_threshold=dynaface.facial.DEFAULT_TILT_THRESHOLD,
):
    if stats is None:
        stats = dynaface.measures.all_measures()
    img = dynaface.image.load_image(filename)
    face = dynaface.facial.AnalyzeFace(stats, tilt_threshold=tilt_threshold)
    face.load_image(img, crop)
    return face
