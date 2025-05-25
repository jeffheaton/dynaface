import math
import os
from typing import List, Optional, Tuple
from numpy.typing import NDArray
from typing import Any, List, Optional, Tuple

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


def load_image(filename: str) -> NDArray[Any]:
    """
    Load an image from a file and convert it to RGB format.

    Args:
        filename (str): Path to the image file.

    Returns:
        NDArray[Any]: The loaded image in RGB format.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(os.path.abspath(filename))
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ImageAnalysis:

    def __init__(self) -> None:
        self.text_font: int = cv2.FONT_HERSHEY_SIMPLEX
        self.text_size: float = 0.75
        self.text_color: Tuple[int, int, int] = (255, 255, 255)
        self.text_thick: int = 2
        self.text_back: int = 5
        self.stats_right: int = 750
        self.width: int = 0
        self.height: int = 0
        self.shape: Tuple[int, ...] = (0, 0, 3)

    def _check_image(self) -> None:
        """
        Check if the image is loaded.

        Raises:
            ValueError: If the image is not loaded.
        """
        if not self.is_image_loaded():
            raise ValueError("Image not loaded. Please load an image first.")

    def is_image_loaded(self) -> bool:
        """
        Check if an image is loaded.

        Returns:
            bool: True if an image is loaded, False otherwise.
        """
        return hasattr(self, "render_img")

    def load_image(self, img: NDArray[Any]) -> bool:
        """
        Load an image into the analysis class.

        Args:
            img (NDArray[Any]): Input image.
        Returns:
            bool: True if the image was processed, False otherwise.
        Raises:
            TypeError: If img is None.
            ValueError: If img is too small.
        """

        if img.shape[0] < 5 or img.shape[1] < 5:
            raise ValueError("Image is empty")

        self.init_image(img)
        return True

    def init_image(self, img: NDArray[Any]) -> None:
        """
        Initialize the image and related attributes.

        Args:
            img (NDArray[Any]): Input image.
        """
        self.original_img: NDArray[Any] = img.copy()
        self.gray_img: NDArray[Any] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.render_img: NDArray[Any] = img.copy()
        self.original_hsv: NDArray[Any] = cv2.cvtColor(
            self.original_img, cv2.COLOR_RGB2HSV
        ).astype(np.int64)
        self.shape = self.original_img.shape

        self.height, self.width = self.original_img.shape[:2]

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
        self._check_image()
        size = self.text_size * size
        thick = int(self.text_thick * thick)
        textSize, baseline = cv2.getTextSize(txt, self.text_font, size, thick)
        width, height = textSize
        return ((width, height), baseline)

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
        self._check_image()
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
        self._check_image()
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
        self._check_image()
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
        self._check_image()
        if radius is None:
            radius = int(self.render_img.shape[0] // 200)
        cv2.circle(self.render_img, pt, radius, color, -1)

    def render_reset(self) -> None:
        """
        Reset the render image to the original image.
        """
        self._check_image()
        # self.render_img = self.original_img.copy()
        self.render_img[:, :] = self.original_img

    def extract_horiz(
        self, y: int, x1: Optional[int] = None, x2: Optional[int] = None
    ) -> NDArray[Any]:
        """
        Extract a horizontal section of the image.

        Args:
            y (int): Row index.
            x1 (Optional[int]): Start column index.
            x2 (Optional[int]): End column index.

        Returns:
            NDArray[Any]: Extracted image section.
        """
        self._check_image()
        if not x1:
            x1 = 0
        if not x2:
            x2 = self.render_img.shape[0]
        return self.original_img[y, x1:x2]

    def extract_horiz_hsv(
        self, y: int, x1: Optional[int] = None, x2: Optional[int] = None
    ) -> NDArray[Any]:
        """
        Extract a horizontal section of the HSV image.

        Args:
            y (int): Row index.
            x1 (Optional[int]): Start column index.
            x2 (Optional[int]): End column index.

        Returns:
            NDArray[Any]: Extracted image section in HSV.
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
        self._check_image()
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
        self._check_image()
        # Create an integer numpy array for rendering.
        contours_arr = np.array(contours, dtype=np.int32)

        if render:
            overlay = self.render_img.copy()
            cv2.fillPoly(overlay, pts=[contours_arr], color=color)
            self.render_img[:, :] = cv2.addWeighted(
                overlay, alpha, self.render_img, 1 - alpha, 0
            ).astype(self.render_img.dtype)

        # Create a separate float array for area measurement.
        scaled_contours = np.array(contours, dtype=np.float64) * pix2mm
        x = scaled_contours[:, 0]
        y = scaled_contours[:, 1]

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
        self._check_image()
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
        self._check_image()
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

    def rectangle(
        self,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
        filled: bool = False,
        alpha: float = 1.0,
    ) -> None:
        """
        Draws a rectangle on the image.

        Args:
            top_left (Tuple[int, int]): Top-left corner of the rectangle.
            bottom_right (Tuple[int, int]): Bottom-right corner of the rectangle.
            color (Tuple[int, int, int]): Color of the rectangle (default is red).
            thickness (int): Thickness of the rectangle border.
            filled (bool): Whether the rectangle is filled or not.
            alpha (float): Opacity level of filled rectangle (ignored if not filled).
        """
        self._check_image()

        if filled:
            overlay = self.render_img.copy()
            cv2.rectangle(overlay, top_left, bottom_right, color, thickness=cv2.FILLED)
            cv2.addWeighted(
                overlay, alpha, self.render_img, 1 - alpha, 0, self.render_img
            )
        else:
            cv2.rectangle(self.render_img, top_left, bottom_right, color, thickness)

    def sample_rectangle(
        self,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        as_hsv: bool = False,
    ) -> NDArray[Any]:
        """
        Extract RGB or HSV pixel values from a rectangular region.

        Args:
            top_left (Tuple[int, int]): Top-left corner of the rectangle.
            bottom_right (Tuple[int, int]): Bottom-right corner of the rectangle.
            as_hsv (bool): If True, returns HSV values; otherwise, returns RGB values.

        Returns:
            NDArray[Any]: Array of pixel values with shape (n, 3), where n is the number of pixels.
                        If the region is empty or invalid, returns an empty array of shape (0, 3).
        """
        self._check_image()

        x1, y1 = top_left
        x2, y2 = bottom_right

        # Ensure coordinates are within image bounds
        x1, x2 = sorted((max(0, x1), min(self.width, x2)))
        y1, y2 = sorted((max(0, y1), min(self.height, y2)))

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            logger.warning(
                f"Invalid or empty rectangle: top_left={top_left}, bottom_right={bottom_right}, "
                f"clamped to x=({x1},{x2}), y=({y1},{y2})"
            )
            return np.empty((0, 3), dtype=np.uint8)

        # Extract rectangle region
        region = self.original_img[y1:y2, x1:x2, :]

        if region.size == 0:
            logger.warning(f"Sampled region is empty: x=({x1},{x2}), y=({y1},{y2})")
            return np.empty((0, 3), dtype=np.uint8)

        if as_hsv:
            region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)

        # Flatten the region into an (n, 3) array
        pixels = region.reshape(-1, 3)

        return pixels
