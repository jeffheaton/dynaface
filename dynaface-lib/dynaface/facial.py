import copy
import logging
import math
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
from dynaface import measures, models, util
from dynaface.image import ImageAnalysis, load_image
from dynaface.lateral import analyze_lateral
from dynaface.spiga.inference.config import ModelConfig
from dynaface.spiga.inference.framework import SPIGAFramework

STYLEGAN_WIDTH = 1024
STYLEGAN_LEFT_PUPIL = (640, 480)
STYLEGAN_RIGHT_PUPIL = (380, 480)
STYLEGAN_PUPIL_DIST = STYLEGAN_LEFT_PUPIL[0] - STYLEGAN_RIGHT_PUPIL[0]

STD_PUPIL_DIST = 63
DEFAULT_TILT_THRESHOLD = -1

LM_LEFT_PUPIL = 97
LM_RIGHT_PUPIL = 96

FILL_COLOR = [255, 255, 255]


SPIGA_MODEL = "wflw"

logger = logging.getLogger(__name__)


def init_processor(device=None):
    global _processor

    if not device:
        has_mps = torch.backends.mps.is_built()
        device = "mps" if has_mps else "gpu" if torch.cuda.is_available() else "cpu"
        # device = "cpu"

    config = ModelConfig(dataset_name=SPIGA_MODEL, load_model_url=False)
    _processor = SPIGAFramework(config, device=device)


def util_calc_pd(pupils):
    left_pupil, right_pupil = pupils
    left_pupil = np.array(left_pupil)
    right_pupil = np.array(right_pupil)

    # Calculate Euclidean distance between the two pupils
    pupillary_distance = np.linalg.norm(left_pupil - right_pupil)

    # Convert the distance from pixels to millimeters
    pix2mm = AnalyzeFace.pd / pupillary_distance

    return pupillary_distance, pix2mm


def util_get_pupils(landmarks):
    return landmarks[LM_LEFT_PUPIL], landmarks[LM_RIGHT_PUPIL]


class AnalyzeFace(ImageAnalysis):
    pd = STD_PUPIL_DIST

    def __init__(self, measures=None, tilt_threshold=DEFAULT_TILT_THRESHOLD):
        self.original_img = None
        self.left_eye = None
        self.right_eye = None
        self.nose = None
        self._headpose = None
        self.flipped = False
        if measures is None:
            self.measures = measures.all_measures()
        else:
            self.measures = measures
        self.headpose = [0, 0, 0]
        self.landmarks = []
        self.lateral = False
        self.lateral_landmarks = np.full((6, 2), -1.0)
        self.pupillary_distance = 0
        logger.debug(f"===INIT: t={tilt_threshold}")
        self.tilt_threshold = tilt_threshold
        self.pix2mm = 1
        self.face_rotation = None
        self.orig_pupils = None

    def get_all_items(self):
        return [
            stat.name
            for obj in self.measures
            if obj.enabled
            for stat in obj.items
            if stat.enabled
        ]

    def _find_landmarks(self, img):
        logger.debug("Called _find_landmarks")
        start_time = time.time()
        bbox, prob = models.mtcnn_model.detect(img)

        if prob[0] == None or prob[0] < 0.9:
            return None, None

        end_time = time.time()
        mtcnn_duration = end_time - start_time
        logger.debug(f"Detected bbox: {bbox}")

        if bbox is None:
            bbox = [0, 0, img.shape[1], img.shape[0]]
            logging.info("MTCNN could not detect face area")
        else:
            bbox = bbox[0]
        # bbox to spiga is x,y,w,h; however, facenet_pytorch deals in x1,y1,x2,y2.

        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        logger.debug("Calling SPIGA")
        start_time = time.time()
        features = models.spiga_model.inference(img, [bbox])
        end_time = time.time()
        spiga_duration = end_time - start_time

        logger.debug(
            f"Elapsed time (sec): MTCNN={mtcnn_duration:,}, SPIGA={spiga_duration:,}"
        )
        # Prepare variables
        x0, y0, w, h = bbox
        landmarks2 = models.convert_landmarks(features)[0]

        headpose = np.array(features["headpose"][0])
        return landmarks2, headpose

    def is_lateral(self) -> Tuple[bool, bool]:
        """
        Determines whether the head pose is lateral and whether the head is facing left.

        Returns:
            Tuple[bool, bool]:
            - First value (bool): True if the head is in a lateral pose (yaw angle beyond ±15 degrees), False otherwise.
            - Second value (bool): True if the head is facing left (yaw < 0), False if facing right or frontal.

        If `_headpose` is None, it defaults to (False, False).
        """
        if self._headpose is None:
            return False, False  # Default when head pose data is unavailable

        yaw, pitch, roll = self._headpose[:3]  # Extract yaw, pitch, and roll values
        logger.info(f"Headpose: yaw:{yaw}, pitch:{pitch}, roll:{roll}")

        is_lateral: bool = abs(yaw) > 15  # Lateral if yaw exceeds ±15 degrees
        is_facing_left: bool = (
            yaw < 0
        )  # True if facing left, False if facing right or frontal

        return is_lateral, is_facing_left

    def _overlay_lateral_analysis(self, c):
        """Scales and overlays the lateral analysis image onto self.render_img at the top-right."""
        if c is None:
            return

        # Scale 'c' to a height of 1024 while maintaining aspect ratio
        c_height, c_width = c.shape[:2]
        scale_factor = 1024 / c_height

        new_width = int(c_width * scale_factor)
        c_resized = cv2.resize(c, (new_width, 1024))

        MAX_INSERT_WIDTH = 1024  # Disabled currently, as 1024 is max

        # Limit width to a maximum of MAX_INSERT_WIDTH
        if new_width > MAX_INSERT_WIDTH:
            new_width = MAX_INSERT_WIDTH  # Crop to MAX_INSERT_WIDTH max width
            c_resized = c_resized[:, :MAX_INSERT_WIDTH]  # Keep the left side

        # Overlay the resized image onto self.render_img at the top-right corner
        render_h, render_w = self.render_img.shape[:2]

        # Define the position at the top-right corner
        x_offset = render_w - new_width
        y_offset = 0

        # Blend the images
        self.render_img[y_offset : y_offset + 1024, x_offset : x_offset + new_width] = (
            c_resized
        )
        super().load_image(self.render_img)

    def load_image(self, img, crop, pupils=None):
        super().load_image(img)
        logger.debug("Low level-image loaded")
        self.landmarks, self._headpose = self._find_landmarks(img)

        lateral_pos, facing_left = self.is_lateral()

        if lateral_pos and self.landmarks:
            self.lateral = True
            self.pix2mm = 0.24
            if not facing_left:
                self.flipped = True
                flipped = cv2.flip(self.original_img, 1)
                super().load_image(flipped)
                self.landmarks, self._headpose = self._find_landmarks(flipped)
            else:
                self.flipped = False

            self.crop_lateral()
        elif self.landmarks is not None:
            logger.debug("Landmarks located")
            self.calc_pd()
            self.lateral = False

            if crop:
                logger.debug("Cropping")
                self.crop_stylegan(pupils=pupils)
        else:
            logger.info("No face detected")
            return False

        if lateral_pos:
            p = util.cv2_to_pil(self.render_img)
            c, self.lateral_landmarks, self.sagittal_x, self.sagittal_y = (
                analyze_lateral(p)
            )
            c = util.trim_sides(c)
            # cv2.imwrite("debug_overlay.png", c)
            self._overlay_lateral_analysis(c)

        return True

    def draw_landmarks(self, size=0.25, color=[0, 255, 255], numbers=False):
        if self.landmarks is None:
            return
        for i, landmark in enumerate(self.landmarks):
            self.circle(landmark, radius=3, color=color)
            if numbers:
                self.write_text([landmark[0] + 3, landmark[1]], str(i), size=0.5)
        self.circle(self.left_eye, color=color)
        self.circle(self.right_eye, color=color)

    def measure(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
        render: bool = True,
        dir: str = "r",
    ) -> float:
        """
        Measures the Euclidean distance between two points (pt1 and pt2) and optionally renders an arrow and text.

        Parameters:
        - `pt1` (Tuple[int, int]): First point (x, y).
        - `pt2` (Tuple[int, int]): Second point (x, y).
        - `color` (Tuple[int, int, int], optional): Color of the rendered arrow. Default is red (255, 0, 0).
        - `thickness` (int, optional): Thickness of the arrow line. Default is 3.
        - `render` (bool, optional): Whether to render the measurement visually. Default is True.
        - `dir` (str, optional): Direction for placing the text label ('r' for right, else left). Default is 'r'.

        Returns:
        - `float`: The measured distance in millimeters.
        """

        # If rendering is enabled, draw an arrow between the two points
        if render:
            self.arrow(pt1, pt2, color, thickness)

        # Compute Euclidean distance and convert pixels to millimeters
        d: float = math.dist(pt1, pt2) * self.pix2mm

        # Format measurement text
        txt: str = f"{d:.2f}mm"

        # Calculate text size for positioning
        m: Tuple[Tuple[int, int], int] = self.calc_text_size(txt)

        # Compute the midpoint for displaying text
        if dir == "r":
            mp: List[int] = [
                int((pt1[0] + pt2[0]) // 2) + 15,
                int((pt1[1] + pt2[1]) // 2),
            ]
        elif dir == "a":
            mp: List[int] = [
                int(min(pt1[0], pt2[0]) + 15),
                int((pt1[1] + pt2[1]) // 2) - 20,
            ]
        else:
            mp: List[int] = [
                int(pt1[0] + 15),
                int((pt1[1] + pt2[1]) // 2),
            ]

        # Render text if enabled
        if render:
            self.write_text(mp, txt)

        return d

    def measure_curve(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        sagittal_x: np.ndarray,
        sagittal_y: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
        render: bool = True,
        dir: str = "r",
    ) -> float:
        """
        Measures the curved distance along a sagittal line between two points (pt1 and pt2) and optionally renders the curve and text.

        Parameters:
        - `pt1` (Tuple[int, int]): First point (x, y).
        - `pt2` (Tuple[int, int]): Second point (x, y).
        - `sagittal_x` (np.ndarray): Array of x-coordinates forming the curve.
        - `sagittal_y` (np.ndarray): Array of y-coordinates forming the curve.
        - `color` (Tuple[int, int, int], optional): Color of the rendered curve. Default is red (255, 0, 0).
        - `thickness` (int, optional): Thickness of the curve line. Default is 3.
        - `render` (bool, optional): Whether to render the measurement visually. Default is True.
        - `dir` (str, optional): Direction for placing the text label ('r' for right, else left). Default is 'r'.

        Returns:
        - `float`: The measured curved distance in millimeters.
        """

        # Combine sagittal_x and sagittal_y into a NumPy array of (x, y) points
        sagittal_line = np.column_stack((sagittal_x, sagittal_y))

        # Function to find closest index for a given point
        def find_closest_index(point, line):
            distances = np.linalg.norm(line - np.array(point), axis=1)
            return np.argmin(distances)

        # Find the closest indices in the sagittal line for pt1 and pt2
        idx1 = find_closest_index(pt1, sagittal_line)
        idx2 = find_closest_index(pt2, sagittal_line)

        # Ensure correct ordering
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        # Extract the segment of the curve between pt1 and pt2
        segment = sagittal_line[idx1 : idx2 + 1]

        # Compute total curved distance along the segment
        d = (
            sum(math.dist(segment[i], segment[i + 1]) for i in range(len(segment) - 1))
            * self.pix2mm
        )

        # Format measurement text
        txt = f"{d:.2f}mm"

        # Compute the midpoint for displaying text
        mid_idx = len(segment) // 2
        mp = (
            (segment[mid_idx][0] + 15, segment[mid_idx][1])
            if dir == "r"
            else (segment[mid_idx][0] - 15, segment[mid_idx][1])
        )

        # Render curve and text if enabled
        if render:
            self.draw_curve(segment, color, thickness)
            if hasattr(self, "write_text"):
                self.write_text(mp, txt)

        return d

    def draw_curve(
        self, segment: np.ndarray, color: Tuple[int, int, int], thickness: int
    ) -> None:
        """
        Draws a curve connecting a segment of points.

        Parameters:
        - `segment` (np.ndarray): List of (x, y) points to draw.
        - `color` (Tuple[int, int, int]): Color of the curve.
        - `thickness` (int): Thickness of the curve line.
        """
        if len(segment) < 2:
            return  # Not enough points to draw a curve

        # Convert points to integer format required for OpenCV
        curve_pts = segment.astype(np.int32)

        # Draw polyline on the image (assuming `self.image` exists as the frame to draw on)
        cv2.polylines(
            self.render_img,
            [curve_pts],
            isClosed=False,
            color=color,
            thickness=thickness,
        )

    def analyze_next_pt(self, txt):
        result = (self.analyze_x, self.analyze_y)
        m = self.calc_text_size(txt)
        self.analyze_y += int(m[0][1] * 2)
        return result

    def analyze(self):
        if self.landmarks is None:
            return
        m = self.calc_text_size("W")
        self.analyze_x = int(m[0][0] * 0.25)
        self.analyze_y = int(m[0][1] * 1.5)
        result = {}
        for calc in self.measures:
            if calc.enabled:
                result.update(calc.calc(self))
        return result

    def calculate_face_rotation(self):
        p = util_get_pupils(self.landmarks)
        return measures.to_degrees(util.calculate_face_rotation(p))

    def crop_lateral(self):
        INFLATE_LATERAL_TOP = 0.1  # Increase top by 10%
        INFLATE_LATERAL_BOTTOM = 0.1  # Increase bottom by 10%

        bbox, _ = models.mtcnn_model.detect(self.render_img)
        bbox = bbox[0]

        crop_x, crop_y, w, h = (
            int(bbox[0]),  # x-min
            int(bbox[1]),  # y-min
            int(bbox[2] - bbox[0]),  # width
            int(bbox[3] - bbox[1]),  # height
        )

        # Compute vertical expansion amounts
        expand_top = int(h * INFLATE_LATERAL_TOP)
        expand_bottom = int(h * INFLATE_LATERAL_BOTTOM)

        # Ensure new crop_y does not go out of bounds
        crop_y = max(0, crop_y - expand_top)  # Move up by expand_top

        # Adjust height
        h = h + expand_top + expand_bottom

        # Compute aspect ratio and scale to match STYLEGAN_WIDTH
        width, height = self.render_img.shape[1], self.render_img.shape[0]
        ar = width / height
        new_width = STYLEGAN_WIDTH
        new_height = int(new_width / ar)
        scale = new_width / width

        img2 = cv2.resize(self.render_img, (new_width, new_height))

        crop_x = int((self.landmarks[96][0] * scale) - (STYLEGAN_WIDTH * 0.25))
        crop_y = int((self.landmarks[96][1] * scale) - STYLEGAN_RIGHT_PUPIL[1])

        img2, _, _ = util.safe_clip(
            img2,
            crop_x,
            crop_y,
            STYLEGAN_WIDTH,
            STYLEGAN_WIDTH,
            FILL_COLOR,
        )
        self.landmarks = util.scale_crop_points(self.landmarks, crop_x, crop_y, scale)

        # Reload Image
        super().load_image(img2)

    def crop_stylegan(self, pupils=None):
        # Save orig pupils so we can lock the scale, rotate, and crop during a load
        self.orig_pupils = util_get_pupils(self.landmarks)

        pupils = self.orig_pupils if pupils is None else pupils
        # Rotate, if needed
        img2 = self.original_img
        if pupils:
            r = util.calculate_face_rotation(pupils)
            tilt = measures.to_degrees(r)
            if (self.tilt_threshold >= 0) and (abs(tilt) > self.tilt_threshold):
                logger.debug(
                    f"Rotate landmarks: detected tilt={tilt} threshold={self.tilt_threshold}"
                )
                self.face_rotation = r
                center = (
                    self.original_img.shape[1] // 2,
                    self.original_img.shape[0] // 2,
                )
                self.landmarks = util.rotate_crop_points(self.landmarks, center, tilt)
            else:
                self.face_rotation = None

        if not pupils:
            pupils = util_get_pupils(landmarks=self.landmarks)

        d, _ = util_calc_pd(pupils)

        if d == 0:
            raise ValueError("Can't process face pupils must be in different locations")

        if self.face_rotation:
            logger.debug(f"Fix tilt: {self.face_rotation}")
            img2 = util.straighten(self.original_img, self.face_rotation)
        width, height = img2.shape[1], img2.shape[0]

        ar = width / height
        new_width = int(width * (STYLEGAN_PUPIL_DIST / d))
        new_height = int(new_width / ar)
        scale = new_width / width
        crop_x = int((self.landmarks[96][0] * scale) - STYLEGAN_RIGHT_PUPIL[0])
        crop_y = int((self.landmarks[96][1] * scale) - STYLEGAN_RIGHT_PUPIL[1])

        img2 = cv2.resize(img2, (new_width, new_height))

        img2, _, _ = util.safe_clip(
            img2,
            crop_x,
            crop_y,
            STYLEGAN_WIDTH,
            STYLEGAN_WIDTH,
            FILL_COLOR,
        )
        self.landmarks = util.scale_crop_points(self.landmarks, crop_x, crop_y, scale)

        # Reload Image
        super().load_image(img2)

    def calc_pd(self):
        self.pupillary_distance, self.pix2mm = util_calc_pd(self.get_pupils())

    def get_pupils(self):
        return util_get_pupils(self.landmarks)

    def dump_state(self):
        result = [
            self.original_img,
            self.headpose,
            self.landmarks,
            self.pupillary_distance,
            self.pix2mm,
            self.face_rotation,
        ]
        return copy.copy(result)

    def load_state(self, obj):
        if self.original_img is None:
            self.init_image(obj[0])
        else:
            self.original_img = obj[0][:]

        self.headpose = copy.copy(obj[1])
        self.landmarks = copy.copy(obj[2])
        self.pupillary_distance = obj[3]
        self.pix2mm = obj[4]

        try:
            self.face_rotation = obj[5]
        except IndexError:
            self.face_rotation = 0
        except TypeError:
            self.face_rotation = 0

    def find_pupils(self):
        return util_get_pupils(self.landmarks)

    def calc_bisect(self):
        return util.bisecting_line_coordinates(img_size=1024, pupils=self.find_pupils())

    def draw_static(self):
        if self.lateral:
            str = "Lateral (right)" if self.flipped else "Lateral (left)"
            self.write_text((10, self.height - 20), str, size=2)
