import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dynaface.util import PolyArea

COLORS = np.array(
    [
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)
CLUST_NUM = 3
USE_HSV = False


def load_image(filename: str) -> np.ndarray:
    """
    Load an image from a file and convert it to RGB format.

    Args:
        filename (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image in RGB format.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(os.path.abspath(filename))
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ImageAnalysis:
    def load_image(self, img: np.ndarray) -> None:
        """
        Load an image into the analysis class.

        Args:
            img (np.ndarray): Input image.

        Raises:
            TypeError: If img is None.
            ValueError: If img is too small.
        """
        if img is None:
            raise TypeError("Expected a value for img, but got None instead.")

        if img.shape[0] < 5 or img.shape[1] < 5:
            raise ValueError("Image is empty")

        self.init_image(img)

    def init_image(self, img: np.ndarray) -> None:
        """
        Initialize the image and related attributes.

        Args:
            img (np.ndarray): Input image.
        """
        self.original_img = img.copy()
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.render_img = img.copy()
        self.original_hsv = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2HSV).astype(
            np.int64
        )
        self.shape = self.original_img.shape

        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_size = 0.75
        self.text_color = (255, 255, 255)
        self.text_thick = 2
        self.text_back = 5

        self.height, self.width = self.original_img.shape[:2]

        self.stats_right = 750

    def write_text(
        self,
        pos: Tuple[int, int],
        txt: str,
        color: Optional[Tuple[int, int, int]] = None,
        size: float = 1,
        thick: int = 1,
    ) -> None:
        """
        Write text on the image.

        Args:
            pos (Tuple[int, int]): Position to place the text.
            txt (str): Text to write.
            color (Optional[Tuple[int, int, int]]): Text color, defaults to white.
            size (float): Scaling factor for text size.
            thick (int): Thickness of the text.
        """
        size = self.text_size * size
        thick = int(self.text_thick * thick)
        if color is None:
            color = self.text_color
        cv2.putText(
            self.render_img,
            txt,
            pos,
            self.text_font,
            size,
            (0, 0, 0),
            thick + self.text_back,
            cv2.LINE_AA,
        )

        cv2.putText(
            self.render_img,
            txt,
            pos,
            self.text_font,
            size,
            self.text_color,
            thick,
            cv2.LINE_AA,
        )

    def calc_text_size(
        self, txt: str, size: float = 1, thick: int = 1
    ) -> Tuple[Tuple[int, int], int]:
        """
        Calculate text size and baseline.

        Args:
            txt (str): Text string.
            size (float): Text size scale.
            thick (int): Text thickness.

        Returns:
            Tuple[Tuple[int, int], int]: Text size and baseline.
        """
        size = self.text_size * size
        thick = int(self.text_thick * thick)
        textSize, baseline = cv2.getTextSize(txt, self.text_font, size, thick)
        return (tuple(textSize), baseline)

    def write_text_sq(
        self,
        pos: Tuple[int, int],
        txt: str,
        color: Optional[Tuple[int, int, int]] = None,
        mark: str = "2",
        up: int = 5,
    ) -> None:
        """
        Write text with a square marker on the image.

        Args:
            pos (Tuple[int, int]): Position to place the text.
            txt (str): Text to write.
            color (Optional[Tuple[int, int, int]]): Text color.
            mark (str): Marker character.
            up (int): Upward offset for the marker.
        """
        if color is None:
            color = self.text_color

        cv2.putText(
            self.render_img,
            txt,
            pos,
            self.text_font,
            self.text_size,
            (0, 0, 0),
            self.text_thick + self.text_back,
            cv2.LINE_AA,
        )
        w1 = cv2.getTextSize(
            txt, self.text_font, self.text_size, self.text_thick + self.text_back
        )[0][0]

        cv2.putText(
            self.render_img,
            txt,
            pos,
            self.text_font,
            self.text_size,
            self.text_color,
            self.text_thick,
            cv2.LINE_AA,
        )
        cv2.getTextSize(txt, self.text_font, self.text_size, self.text_thick)[0][0]

        cv2.putText(
            self.render_img,
            mark,
            (pos[0] + w1 - 5, pos[1] - up),
            self.text_font,
            self.text_size,
            (0, 0, 0),
            self.text_thick + self.text_back,
            cv2.LINE_AA,
        )

        cv2.putText(
            self.render_img,
            mark,
            (pos[0] + w1 - 5, pos[1] - up),
            self.text_font,
            self.text_size,
            self.text_color,
            self.text_thick,
            cv2.LINE_AA,
        )

    def hline(
        self,
        y: int,
        x1: Optional[int] = None,
        x2: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 255, 255),
        width: int = 5,
    ) -> None:
        """
        Draw a horizontal line.

        Args:
            y (int): Y-coordinate.
            x1 (Optional[int]): Starting X-coordinate.
            x2 (Optional[int]): Ending X-coordinate.
            color (Tuple[int, int, int]): Line color.
            width (int): Line width.
        """
        if not x1:
            x1 = 0
        if not x2:
            x2 = self.render_img.shape[0]
        cv2.line(self.render_img, (x1, y), (x2, y), color, width)

    def vline(
        self,
        x: int,
        y1: Optional[int] = None,
        y2: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 255, 255),
        width: int = 5,
    ) -> None:
        """
        Draws a vertical line in the image.

        :param x: X-coordinate where the vertical line should be drawn.
        :param y1: Starting Y-coordinate (defaults to the top of the image if None).
        :param y2: Ending Y-coordinate (defaults to the bottom of the image if None).
        :param color: Color of the line in BGR format.
        :param width: Thickness of the line.
        """
        if not y1:
            y1 = 0
        if not y2:
            y2 = self.render_img.shape[1]
        cv2.line(self.render_img, (x, y1), (x, y2), color, width)

    def circle(
        self,
        pt: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: Optional[int] = None,
    ) -> None:
        """
        Draw a circle on the image.

        Args:
            pt (Tuple[int, int]): Center point.
            color (Tuple[int, int, int]): Circle color.
            radius (Optional[int]): Circle radius.
        """
        if radius is None:
            radius = int(self.render_img.shape[0] // 200)
        cv2.circle(self.render_img, pt, radius, color, -1)

    def render_reset(self) -> None:
        """
        Reset the render image to the original image.
        """
        # self.render_img = self.original_img.copy()
        self.render_img[:, :] = self.original_img

    def extract_horiz(
        self, y: int, x1: Optional[int] = None, x2: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract a horizontal section of the image.

        Args:
            y (int): Row index.
            x1 (Optional[int]): Start column index.
            x2 (Optional[int]): End column index.

        Returns:
            np.ndarray: Extracted image section.
        """
        if not x1:
            x1 = 0
        if not x2:
            x2 = self.render_img.shape[0]
        return self.original_img[y, x1:x2]

    def extract_horiz_hsv(
        self, y: int, x1: Optional[int] = None, x2: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract a horizontal section of the HSV image.

        Args:
            y (int): Row index.
            x1 (Optional[int]): Start column index.
            x2 (Optional[int]): End column index.

        Returns:
            np.ndarray: Extracted image section in HSV.
        """
        if not x1:
            x1 = 0
        if not x2:
            x2 = self.render_img.shape[0]
        return self.original_hsv[y, x1:x2]

    def save(self, filename: str) -> None:
        """
        Saves the rendered image to a file.

        :param filename: The path where the image should be saved.
        """
        image = cv2.cvtColor(self.render_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image)

    def measure_polygon(
        self,
        contours: List[Tuple[int, int]],
        pix2mm: float,
        alpha: float = 0.4,
        color: Tuple[int, int, int] = (0, 0, 255),
        render: bool = True,
    ) -> float:
        """
        Measures the area of a polygon in the image.

        :param contours: A list of points defining the polygon.
        :param pix2mm: Conversion factor from pixels to millimeters.
        :param alpha: Opacity for rendering overlay.
        :param color: Color for the polygon overlay.
        :param render: If True, renders the polygon overlay.
        :return: The computed area of the polygon.
        """
        # Create a numpy array from the list with a specified dtype.
        contours_arr = np.array(contours, dtype=np.int32)
        if render:
            overlay = self.render_img.copy()
            cv2.fillPoly(overlay, pts=[contours_arr], color=color)
            self.render_img[:, :] = cv2.addWeighted(
                overlay, alpha, self.render_img, 1 - alpha, 0
            )
        # Multiply the array by pix2mm and extract columns.
        contours_arr = contours_arr * pix2mm
        x = contours_arr[:, 0]
        y = contours_arr[:, 1]
        # Cast the result of PolyArea to float.
        return float(PolyArea(x, y))

    def line(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
    ) -> None:
        """
        Draws a line between two points.

        :param pt1: Starting point of the line.
        :param pt2: Ending point of the line.
        :param color: Color of the line in BGR format.
        :param thickness: Thickness of the line.
        """
        cv2.line(self.render_img, pt1, pt2, color, thickness)

    def arrow_head(
        self, pt1: Tuple[int, int], pt2: Tuple[int, int], par: int = 15
    ) -> None:
        """
        Draws an arrowhead at the end of a line.

        :param pt1: The base point of the arrowhead.
        :param pt2: The tip of the arrow.
        :param par: Size of the arrowhead.
        """
        # https://www.codeguru.com/multimedia/drawing-an-arrowline/
        slopy = math.atan2((pt1[1] - pt2[1]), (pt1[0] - pt2[0]))
        cosy = math.cos(slopy)
        siny = math.sin(slopy)

        self.line(
            pt1,
            (
                pt1[0] + int(-par * cosy - (par / 2.0 * siny)),
                pt1[1] + int(-par * siny + (par / 2.0 * cosy)),
            ),
        )
        self.line(
            pt1,
            (
                pt1[0] + int(-par * cosy + (par / 2.0 * siny)),
                pt1[1] - int(par / 2.0 * cosy + par * siny),
            ),
        )

    def arrow(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
        apt1: bool = True,
        apt2: bool = True,
    ) -> None:
        """
        Draws a line with arrowheads at one or both ends.

        :param pt1: Starting point of the line.
        :param pt2: Ending point of the line.
        :param color: Color of the line and arrowhead.
        :param thickness: Thickness of the line.
        :param apt1: If True, draws an arrowhead at pt1.
        :param apt2: If True, draws an arrowhead at pt2.
        """
        self.line(pt1, pt2, color, thickness)

        if apt1:
            self.arrow_head(pt1, pt2)

        if apt2:
            self.arrow_head(pt2, pt1)
