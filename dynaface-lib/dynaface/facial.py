import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urlparse

import cv2
from dynaface import util
import dynaface.image
import dynaface.measures
from dynaface import config
import numpy as np
import requests  # You may also use: # type: ignore[import]
from dynaface.const import (
    DEFAULT_TILT_THRESHOLD,
    FILL_COLOR,
    LM_LEFT_PUPIL,
    LM_RIGHT_PUPIL,
    STD_PUPIL_DIST,
    STYLEGAN_PUPIL_DIST,
    STYLEGAN_RIGHT_PUPIL,
    STYLEGAN_WIDTH,
)
from dynaface.image import ImageAnalysis
from dynaface.measures import MeasureBase
from dynaface.models import are_models_init
from dynaface.util import VERIFY_CERTS
from numpy.typing import NDArray

import dynaface
from dynaface import measures, models, util

logger = logging.getLogger(__name__)


def util_calc_pd(
    pupils: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[float, float]:
    left_pupil = np.array(pupils[0])
    right_pupil = np.array(pupils[1])

    pupillary_distance = np.linalg.norm(left_pupil - right_pupil)
    pix2mm = AnalyzeFace.pd / pupillary_distance

    return float(pupillary_distance), float(pix2mm)


def util_get_pupils(
    landmarks: List[Tuple[int, int]],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return landmarks[LM_LEFT_PUPIL], landmarks[LM_RIGHT_PUPIL]


class AnalyzeFace(ImageAnalysis):
    pd = STD_PUPIL_DIST

    def __init__(
        self,
        measures: Optional[List[MeasureBase]] = None,
        tilt_threshold: float = DEFAULT_TILT_THRESHOLD,
    ) -> None:
        """
        Initialize the AnalyzeFace object.

        Args:
            measures (Optional[List[MeasureBase]]): Facial measures to be used. Defaults to None.
            tilt_threshold (float): Maximum allowable tilt threshold. Defaults to DEFAULT_TILT_THRESHOLD.
        """
        super().__init__()

        self.render_img: NDArray[np.uint8] = np.zeros((1024, 1024, 3), dtype=np.uint8)
        self.left_eye: Optional[Tuple[int, int]] = None
        self.right_eye: Optional[Tuple[int, int]] = None
        self.nose: Tuple[int, int] = (0, 0)
        self._headpose: NDArray[np.float64] = np.array([0.0, 0.0, 0.0])
        self.flipped: bool = False
        if measures is None:
            self.measures: List[MeasureBase] = dynaface.measures.all_measures()
        else:
            self.measures = measures
        self.headpose: List[int] = [0, 0, 0]
        self.landmarks: List[Tuple[int, int]] = []
        self.lateral: bool = False
        self.lateral_landmarks: NDArray[Any] = NDArray[Any]([])
        self.pupillary_distance: float = 0.0
        self.tilt_threshold: float = tilt_threshold
        self.pix2mm: float = 1.0
        # Changed face_rotation to Optional[float] to allow assigning None.
        self.face_rotation: Optional[float] = 0.0
        self.orig_pupils: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0))
        self.yaw, self.pitch, self.roll = 0.0, 0.0, 0.0

    def get_all_items(self) -> List[str]:
        return [
            stat.name
            for obj in self.measures
            if obj.enabled
            for stat in obj.items
            if stat.enabled
        ]

    def is_no_face(self) -> bool:
        """
        Check if no face is detected.

        Returns:
            bool: True if no face is detected, False otherwise.
        """
        return len(self.landmarks) == 0

    def _find_landmarks(
        self, img: NDArray[Any]
    ) -> Tuple[List[Tuple[int, int]], NDArray[Any]]:
        logger.debug("Called _find_landmarks")
        start_time = time.time()

        if not are_models_init():
            raise ValueError(
                "Models not initialized, please call dynaface.models.init_models()"
            )

        # Ensure the mtcnn model is available.
        if models.mtcnn_model is None:
            raise ValueError("MTCNN model not initialized, please call init_models()")

        bbox, prob = models.mtcnn_model.detect(img)  # type: ignore

        if prob[0] is None or prob[0] < 0.9:
            # Return an ndarray for headpose instead of a list.
            return [], np.array([0, 0, 0])

        end_time = time.time()
        mtcnn_duration = end_time - start_time
        logger.debug(f"Detected bbox: {bbox}")

        if bbox is None:
            bbox = [0, 0, img.shape[1], img.shape[0]]
            logging.info(
                "MTCNN could not detect face area, passing entire image to SPIGA"
            )
        else:
            bbox = bbox[0]
        # Convert bbox from x1,y1,x2,y2 to x,y,w,h
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        logger.debug("Calling SPIGA")
        start_time = time.time()
        # Ensure the spiga model is available.
        assert models.spiga_model is not None, "spiga_model is None"
        features = models.spiga_model.inference(img, [bbox])  # type: ignore
        end_time = time.time()
        spiga_duration = end_time - start_time

        logger.debug(
            f"Elapsed time (sec): MTCNN={mtcnn_duration:,}, SPIGA={spiga_duration:,}"
        )
        # Prepare variables
        landmarks2 = models.convert_landmarks(features)[0]

        headpose = np.array(features["headpose"][0])
        return landmarks2, headpose

    def _is_lateral(self) -> Tuple[bool, bool]:
        """
        Determines whether the head pose is lateral and whether the head is
        facing left.

        Returns:
            Tuple[bool, bool]:
            - First value (bool): True if the head is in a lateral pose
              (yaw angle beyond ±20 degrees), False otherwise.
            - Second value (bool): True if the head is facing left
              (yaw < 0), False if facing right or frontal.

        If _headpose is None, it defaults to (False, False).
        """

        if not config.AUTO_LATERAL:
            logger.debug("AUTO_LATERAL is False, skipping lateral detection.")
            return False, False  # Do not detect lateral if AUTO_LATERAL is False

        if self.is_no_face():
            return False, False  # Default when head pose data is unavailable

        # Extract yaw, pitch, and roll values
        yaw, pitch, roll = self._headpose[:3]
        logger.info(f"Headpose: yaw:{yaw}, pitch:{pitch}, roll:{roll}")
        nose_distance1 = self.landmarks[54][0] - self.landmarks[6][0]
        nose_distance2 = self.landmarks[26][0] - self.landmarks[54][0]
        nose_distance = min(nose_distance1, nose_distance2, key=abs)
        nose_distance_ratio = nose_distance / self.shape[1]

        if abs(yaw) > 20 and nose_distance_ratio < 0:
            # Lateral if yaw exceeds ±20 degrees and nose distance ratio is negative
            is_lateral: bool = True
            logger.info(
                f"Detected lateral head pose: yaw={yaw}>20, nose_distance_ratio={nose_distance_ratio}<0"
            )
        else:
            is_lateral: bool = False

        is_facing_left: bool = yaw < 0  # True if facing left

        return is_lateral, is_facing_left

    def _overlay_lateral_analysis(self, c: Optional[NDArray[np.uint8]]) -> None:
        """
        Scales and overlays the lateral analysis image onto self.render_img
        at the top-right.
        """
        if c is None:
            return

        if self.is_no_face():
            return  # Changed from "return False" (invalid for a None-return function)

        # Scale 'c' to a height of 1024 while maintaining aspect ratio
        c_height, c_width = c.shape[:2]
        scale_factor = 1024 / c_height

        new_width = int(c_width * scale_factor)
        c_resized = cv2.resize(c, (new_width, 1024))

        MAX_INSERT_WIDTH = 1024  # Disabled currently, as 1024 is max

        if new_width > MAX_INSERT_WIDTH:
            new_width = MAX_INSERT_WIDTH  # Crop to MAX_INSERT_WIDTH max width
            c_resized = c_resized[:, :MAX_INSERT_WIDTH]

        _, render_w = self.render_img.shape[:2]
        x_offset = render_w - new_width
        y_offset = 0

        self.render_img[y_offset : y_offset + 1024, x_offset : x_offset + new_width] = (
            c_resized
        )

        super().load_image(self.render_img)

    def load_image(
        self,
        img: NDArray[Any],
        crop: Optional[bool] = True,
        pupils: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        force_frontal: Optional[bool] = False,
    ) -> bool:
        """
        Load an image and process facial landmarks.

        Args:
            img (NDArray[Any]): The image to load.
            crop (bool): Whether to crop the face.
            pupils (Optional[Tuple[Tuple[int, int], Tuple[int, int]]]): Optional pupils coordinates.
        Returns:
            bool: True if the image was processed, False otherwise.
        """
        super().load_image(img)
        logger.debug("Low level-image loaded")
        landmarks, self._headpose = self._find_landmarks(img)
        self.yaw, self.pitch, self.roll = self._headpose[:3]
        self.landmarks = [(int(x), int(y)) for x, y in landmarks]

        if force_frontal:
            lateral_pos = False
            facing_left = False
        else:
            lateral_pos, facing_left = self._is_lateral()

        if lateral_pos and self.landmarks:
            self.lateral = True
            self.pix2mm = 0.24
            if not facing_left:
                self.flipped = True
                flipped = cv2.flip(self.original_img, 1)
                super().load_image(flipped)
                landmarks, self._headpose = self._find_landmarks(flipped)
                self.landmarks = landmarks
            else:
                self.flipped = False

            self.crop_lateral()
        elif (
            self.landmarks
        ):  # Changed check from "is not None" since landmarks is always a list.
            logger.debug("Landmarks located")
            self.calc_pd()
            self.lateral = False

            if crop:
                logger.debug("Cropping")
                self.crop_stylegan(pupils=pupils)

        if lateral_pos:
            from dynaface.lateral import analyze_lateral

            p = util.cv2_to_pil(self.render_img)
            # Convert lateral_landmarks and sagittal data from analyze_lateral.
            c, self.lateral_landmarks, self.sagittal_x, self.sagittal_y = (
                analyze_lateral(p, self.landmarks)
            )
            c = util.trim_sides(c)
            # cv2.imwrite("debug_overlay.png", c)
            self._overlay_lateral_analysis(c)

        return True

    def draw_landmarks(
        self,
        size: float = 0.25,
        color: Tuple[int, int, int] = (0, 255, 255),
        numbers: bool = False,
    ) -> None:
        """
        Draws landmarks and optionally their indices on the image.

        Args:
            size (float): Scale size for numbers. Defaults to 0.25.
            color (Tuple[int, int, int]): Color for drawing. Defaults to (0, 255, 255).
            numbers (bool): Whether to draw the landmark index numbers. Defaults to False.
        """
        # Since is_no_face returns a bool, check it directly.
        if self.is_no_face():
            return
        for i, landmark in enumerate(self.landmarks):
            # if i not in (LM_LEFT_PUPIL, LM_RIGHT_PUPIL):
            #    continue
            self.circle((int(landmark[0]), int(landmark[1])), radius=3, color=color)
            if numbers:
                self.write_text(
                    (int(landmark[0]) + 3, int(landmark[1])),
                    str(i),
                    size=0.5,
                    thick=0.5,
                )

        # Changed conditions to check for non-None values.
        if self.left_eye is not None:
            self.circle(self.left_eye, color=color)

        if self.right_eye is not None:
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
        Measures the Euclidean distance between two points (pt1 and pt2) and
        optionally renders an arrow and text.
        """
        if render:
            self.arrow(pt1, pt2, color, thickness)

        d: float = float(math.dist(pt1, pt2) * self.pix2mm)
        txt: str = f"{d:.2f}mm"

        if dir == "r":
            mp = ((pt1[0] + pt2[0]) // 2 + 15, (pt1[1] + pt2[1]) // 2)
        elif dir == "s":
            mp = (pt2[0] + 15, pt2[1])
        elif dir == "a":
            mp = (min(pt1[0], pt2[0]) + 15, (pt1[1] + pt2[1]) // 2 - 20)
        else:
            mp = (pt1[0] + 15, (pt1[1] + pt2[1]) // 2)

        if render:
            self.write_text(mp, txt)

        return d

    def measure_curve(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        sagittal_x: NDArray[Any],
        sagittal_y: NDArray[Any],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
        render: bool = True,
        dir: str = "r",
    ) -> float:
        """
        Measures the curved distance along a sagittal line between two
        points (pt1 and pt2) and optionally renders the curve and text.
        """
        sagittal_line = np.column_stack((sagittal_x, sagittal_y))

        def find_closest_index(point: Tuple[int, int], line: NDArray[Any]) -> int:
            distances = np.linalg.norm(line - np.array(point), axis=1)
            return int(np.argmin(distances))

        idx1 = find_closest_index(pt1, sagittal_line)
        idx2 = find_closest_index(pt2, sagittal_line)

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        segment = sagittal_line[idx1 : idx2 + 1]
        d: float = float(
            sum(math.dist(segment[i], segment[i + 1]) for i in range(len(segment) - 1))
            * self.pix2mm
        )
        txt = f"{d:.2f}mm"
        mid_idx = len(segment) // 2
        if dir == "r":
            mp: Tuple[int, int] = (
                int(segment[mid_idx][0] + 15),
                int(segment[mid_idx][1]),
            )
        else:
            mp = (int(segment[mid_idx][0] - 15), int(segment[mid_idx][1]))
        if render:
            self.draw_curve(segment, color, thickness)
            if hasattr(self, "write_text"):
                self.write_text(mp, txt)
        return d

    def draw_curve(
        self, segment: NDArray[Any], color: Tuple[int, int, int], thickness: int
    ) -> None:
        """
        Draws a curve connecting a segment of points.
        """
        if len(segment) < 2:
            return  # Not enough points to draw a curve

        curve_pts = segment.astype(np.int32)
        cv2.polylines(
            self.render_img,
            [curve_pts],
            isClosed=False,
            color=color,
            thickness=thickness,
        )

    def analyze_next_pt(self, txt: str) -> Tuple[int, int]:
        """
        Determines the next position for analysis text based on current position.
        """
        result = (self.analyze_x, self.analyze_y)
        m = self.calc_text_size(txt)
        self.analyze_y += int(m[0][1] * 2)
        return result

    def analyze(self) -> Optional[Dict[str, Any]]:
        """
        Performs analysis on the face using enabled measures.
        """
        if not self.landmarks:  # Changed check since landmarks is never None.
            return None
        self.width = self.render_img.shape[1]
        self.height = self.render_img.shape[0]
        m = self.calc_text_size("W")
        self.analyze_x = int(m[0][0] * 0.25)
        self.analyze_y = int(m[0][1] * 1.5)
        result: Dict[str, Any] = {}
        for calc in self.measures:
            if calc.enabled:
                # Use type-ignore to bypass missing attribute error on 'calc'
                result.update(calc.calc(self))  # type: ignore[attr-defined]
        return result

    def calculate_face_rotation(self) -> float:
        """
        Calculates the face rotation in degrees.
        """
        p = util_get_pupils(self.landmarks)
        return util.to_degrees(
            util.calculate_face_rotation(
                ((int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])))
            )
        )

    def crop_lateral(self) -> None:
        """
        Crops the image for lateral analysis by inflating the bounding box.
        """
        INFLATE_LATERAL_TOP = 0.1  # Increase top by 10%
        INFLATE_LATERAL_BOTTOM = 0.1  # Increase bottom by 10%

        # Ensure mtcnn_model is available.
        assert models.mtcnn_model is not None, "mtcnn_model is None"
        bbox, _ = models.mtcnn_model.detect(self.render_img)  # type: ignore
        bbox = bbox[0]

        crop_x, crop_y, h = (
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[3] - bbox[1]),
        )

        expand_top = int(h * INFLATE_LATERAL_TOP)
        expand_bottom = int(h * INFLATE_LATERAL_BOTTOM)

        crop_y = max(0, crop_y - expand_top)
        h = h + expand_top + expand_bottom

        width, height = self.render_img.shape[1], self.render_img.shape[0]
        ar = width / height
        new_width = STYLEGAN_WIDTH
        new_height = int(new_width / ar)
        scale = new_width / width

        img2 = cv2.resize(self.render_img, (new_width, new_height))

        crop_x = int((self.landmarks[96][0] * scale) - (STYLEGAN_WIDTH * 0.25))
        crop_y = int((self.landmarks[96][1] * scale) - STYLEGAN_RIGHT_PUPIL[1])

        # Cast the result of safe_clip to ensure non-None types.
        img2, _, _ = util.safe_clip(
            img2,
            crop_x,
            crop_y,
            STYLEGAN_WIDTH,
            STYLEGAN_WIDTH,
            FILL_COLOR,
        )
        # Ensure landmarks is not None by providing a fallback empty list.
        self.landmarks = [
            (int(x), int(y))
            for x, y in util.scale_crop_points(
                [(int(x), int(y)) for x, y in self.landmarks], crop_x, crop_y, scale
            )
        ]
        super().load_image(img2)

    def crop_stylegan(
        self, pupils: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ) -> None:
        """
        Processes and crops the image for StyleGAN.
        """
        pupils = cast(
            Tuple[Tuple[int, int], Tuple[int, int]],
            tuple((int(x), int(y)) for x, y in util_get_pupils(self.landmarks)),
        )
        self.orig_pupils = (
            (int(pupils[0][0]), int(pupils[0][1])) if pupils else (0, 0),
            (int(pupils[1][0]), int(pupils[1][1])) if pupils else (0, 0),
        )
        pupils = self.orig_pupils if util.is_zero_tuple(pupils) else pupils
        img2 = self.original_img
        if pupils:
            r = util.calculate_face_rotation(pupils)
            tilt = util.to_degrees(r)
            if (self.tilt_threshold >= 0) and (abs(tilt) > self.tilt_threshold):
                logger.debug(
                    f"Rotate landmarks: detected tilt={tilt} threshold={self.tilt_threshold}"
                )
                self.face_rotation = r
                center = (
                    self.original_img.shape[1] // 2,
                    self.original_img.shape[0] // 2,
                )
                self.landmarks = util.rotate_crop_points(
                    [(int(x), int(y)) for x, y in self.landmarks], center, tilt
                )
            else:
                self.face_rotation = None
        if util.is_zero_tuple(pupils):
            pupils = util_get_pupils(self.landmarks)
        d, _ = util_calc_pd(pupils)
        if d == 0:
            raise ValueError("Can't process face pupils must be in different locations")

        if self.yaw > 5:
            d = util.correct_distance_2d_for_yaw(d, self.yaw) * 1.3
            logger.debug(f"Corrected pupil distance for yaw={self.yaw}°: {d:.2f}")

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
        img2 = img2
        self.landmarks = [
            (int(x), int(y))
            for x, y in util.scale_crop_points(
                [(int(x), int(y)) for x, y in self.landmarks], crop_x, crop_y, scale
            )
        ]
        super().load_image(img2)

    def calc_pd(self) -> None:
        self.pupillary_distance, self.pix2mm = util_calc_pd(self.get_pupils())

    def get_pupils(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return util_get_pupils(self.landmarks)

    def dump_state(self) -> List[Any]:
        result: List[Any] = [
            self.original_img,
            self.headpose,
            self.landmarks,
            self.pupillary_distance,
            self.pix2mm,
            self.face_rotation,
        ]
        return copy.copy(result)

    def load_state(self, obj: List[Any]) -> None:
        if not self.is_image_loaded():
            self.init_image(obj[0])
        else:
            self.original_img = obj[0][:]
        self.headpose = copy.copy(obj[1])
        self.landmarks = copy.copy(obj[2])
        self.pupillary_distance = obj[3]
        self.pix2mm = obj[4]
        try:
            self.face_rotation = obj[5]
        except (IndexError, TypeError):
            self.face_rotation = 0

    def find_pupils(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return util_get_pupils(self.landmarks)

    def calc_bisect(self) -> Any:
        # Convert pupil coordinates to ints to match the expected type.
        pupils: Tuple[Tuple[int, int], Tuple[int, int]] = (
            (int(self.find_pupils()[0][0]), int(self.find_pupils()[0][1])),
            (int(self.find_pupils()[1][0]), int(self.find_pupils()[1][1])),
        )  # Explicitly type pupils
        return util.bisecting_line_coordinates(img_size=1024, pupils=pupils)

    def draw_static(self) -> None:
        if self.lateral:
            text = "Lateral (right)" if self.flipped else "Lateral (left)"
            self.write_text((10, self.height - 20), text, size=2)


def load_face_image(
    filename: str,
    crop: bool = True,
    measures: Optional[List[MeasureBase]] = None,
    tilt_threshold: float = DEFAULT_TILT_THRESHOLD,
) -> AnalyzeFace:
    """
    Load and analyze a face image from a local file or URL.
    """
    if measures is None:
        measures = dynaface.measures.all_measures()

    parsed = urlparse(filename)
    if parsed.scheme in ("http", "https"):
        response = requests.get(filename, timeout=10, verify=VERIFY_CERTS)
        response.raise_for_status()
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = dynaface.image.load_image(filename)

    face = AnalyzeFace(measures, tilt_threshold=tilt_threshold)
    face.load_image(img, crop)

    return face
