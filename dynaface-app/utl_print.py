from PyQt6.QtGui import QPainter
from PyQt6.QtPrintSupport import QPrintDialog, QPrinter


def print_pixmap(window, pixmap, text=[]):
    printer = QPrinter()
    printDialog = QPrintDialog(printer, window)

    if printDialog.exec():
        painter = QPainter(printer)

        # Get the printer page size in points
        pageRect = printer.pageRect(QPrinter.Unit.Point)

        # Ensure the width is an integer
        pageWidth = int(pageRect.width())

        # Calculate the scale to fit the image within the page width while maintaining aspect ratio
        scale = pageWidth / pixmap.width()
        scaledImageHeight = int(pixmap.height() * scale)

        # Draw the image at the top of the page
        painter.drawPixmap(0, 0, pixmap.scaledToWidth(pageWidth))

        # Calculate the Y-coordinate for the text, placing it just below the image
        textStartY = scaledImageHeight + 10  # Add a small padding below the image

        # Draw the text
        y = textStartY
        for line in text:
            painter.drawText(0, y, line)
            y += 20

        painter.end()
