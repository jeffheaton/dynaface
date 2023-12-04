import logging
import math

import cv2
import numpy as np
import torch
from facial_analysis.calc import *
from facial_analysis.find_face import FindFace
from facial_analysis.image import ImageAnalysis, load_image
from facial_analysis.spiga.inference.config import ModelConfig
from facial_analysis.spiga.inference.framework import SPIGAFramework
from facial_analysis.util import PolyArea

STD_PUPIL_DIST = 63

LM_LEFT_PUPIL = 97
LM_RIGHT_PUPIL = 96

FILL_COLOR = [128, 128, 128]

STYLEGAN_WIDTH = 1024
STYLEGAN_LEFT_PUPIL = (640, 480)
STYLEGAN_RIGHT_PUPIL = (380, 480)
STYLEGAN_PUPIL_DIST = STYLEGAN_LEFT_PUPIL[0] - STYLEGAN_RIGHT_PUPIL[0]

SPIGA_MODEL = "wflw"

STATS = [
    AnalyzeFAI(),
    AnalyzeOralCommissureExcursion(),
    AnalyzeBrows(),
    AnalyzeDentalArea(),
    AnalyzeEyeArea(),
]
_processor = None


def init_processor(device=None):
    global _processor

    if not device:
        has_mps = torch.backends.mps.is_built()
        device = "mps" if has_mps else "gpu" if torch.cuda.is_available() else "cpu"
        # device = "cpu"

    config = ModelConfig(dataset_name=SPIGA_MODEL, load_model_url=False)
    _processor = SPIGAFramework(config, device=device)


def load_face_image(filename, crop=True, stats=STATS, data_path=None, device=None):
    img = load_image(filename)
    face = AnalyzeFace(stats, data_path=data_path, device=device)
    face.load_image(img, crop)
    return face


def safe_clip(cv2_image, x, y, width, height, background):
    """
    Clips a region from an OpenCV image, adjusting for boundaries and filling missing areas with a specified color.

    If the specified region extends beyond the image boundaries, the function adjusts the coordinates and size
    to fit within the image. If the region is partially or completely outside the image, it fills the missing
    areas with the specified background color.

    Args:
    cv2_image (numpy.ndarray): The source image in OpenCV format.
    x (int): The x-coordinate of the top-left corner of the clipping region.
    y (int): The y-coordinate of the top-left corner of the clipping region.
    width (int): The width of the clipping region.
    height (int): The height of the clipping region.
    background (tuple): A tuple (R, G, B) specifying the fill color for missing areas.

    Returns:
    tuple: A tuple containing the clipped image, x offset, and y offset.
           The x and y offsets indicate how much the origin of the clipped image has shifted
           relative to the original image.
    """

    # Image dimensions
    img_height, img_width = cv2_image.shape[:2]

    # Adjust start and end points to be within the image boundaries
    x_start = max(x, 0)
    y_start = max(y, 0)
    x_end = min(x + width, img_width)
    y_end = min(y + height, img_height)

    # Calculate the size of the region that will be clipped from the original image
    clipped_width = x_end - x_start
    clipped_height = y_end - y_start

    # Create a new image filled with the background color
    new_image = np.full((height, width, 3), background, dtype=cv2_image.dtype)

    # Calculate where to place the clipped region in the new image
    new_x_start = max(0, -x)
    new_y_start = max(0, -y)

    # Clip the region from the original image and place it in the new image
    if clipped_width > 0 and clipped_height > 0:
        clipped_region = cv2_image[y_start:y_end, x_start:x_end]
        new_image[
            new_y_start : new_y_start + clipped_height,
            new_x_start : new_x_start + clipped_width,
        ] = clipped_region

    return new_image, new_x_start, new_y_start


def scale_crop_points(lst, crop_x, crop_y, scale):
    lst2 = []
    for pt in lst:
        lst2.append((int(((pt[0] * scale) - crop_x)), int((pt[1] * scale) - crop_y)))
    return lst2


class AnalyzeFace(ImageAnalysis):
    def __init__(self, stats, data_path=None, device=None):
        global _processor

        if device is None:
            has_mps = torch.backends.mps.is_built()
            device = "mps" if has_mps else "gpu" if torch.cuda.is_available() else "cpu"

        if not _processor:
            init_processor(device)

        self.data_path = data_path
        self.left_eye = None
        self.right_eye = None
        self.nose = None
        self.calcs = stats
        self.headpose = [0, 0, 0]
        self.processor = _processor

    def get_all_stats(self):
        return [stat for obj in self.calcs for stat in obj.stats()]

    def _find_landmarks(self, img):
        bbox = FindFace.detect_face(img)

        if bbox is None:
            bbox = [0, 0, img.shape[1], img.shape[0]]
            logging.info("Could not detect face area")
        # bbox to spiga is x,y,w,h; however, facenet_pytorch deals in x1,y1,x2,y2.
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        features = self.processor.inference(img, [bbox])

        # Prepare variables
        x0, y0, w, h = bbox
        landmarks2 = [
            (int(x[0]), int(x[1])) for x in np.array(features["landmarks"][0])
        ]
        headpose = np.array(features["headpose"][0])
        return landmarks2, headpose

    def load_image(self, img, crop, eyes=None):
        super().load_image(img)
        self.landmarks, self._headpose = self._find_landmarks(img)
        self.pupillary_distance = abs(
            self.landmarks[LM_LEFT_PUPIL][0] - self.landmarks[LM_RIGHT_PUPIL][0]
        )
        self.pix2mm = STD_PUPIL_DIST / self.pupillary_distance
        if crop:
            self.crop_stylegan(eyes)

    def draw_landmarks(self, size=0.25, color=[0, 255, 255], numbers=False):
        for i, landmark in enumerate(self.landmarks):
            self.circle(landmark, radius=3, color=color)
            if numbers:
                self.write_text([landmark[0] + 3, landmark[1]], str(i), size=0.5)
        self.circle(self.left_eye, color=color)
        self.circle(self.right_eye, color=color)

    def measure(self, pt1, pt2, color=(255, 0, 0), thickness=3):
        self.arrow(pt1, pt2, color, thickness)
        d = math.dist(pt1, pt2) * self.pix2mm
        mp = [int((pt1[0] + pt2[0]) // 2), int((pt1[1] + pt2[1]) // 2)]
        self.write_text(mp, f"{d:.2f}mm")
        return d

    def analyze(self):
        result = {}
        for calc in self.calcs:
            if calc.enabled:
                result.update(calc.calc(self))
        return result

    def crop_stylegan(self, pupils=None):
        logging.debug(f"Pupils provided: {pupils}")
        if pupils:
            left_eye, right_eye = pupils
        else:
            left_eye, right_eye = self.get_pupils()
            logging.debug(f"Pupils calculated: l:{left_eye}, r:{right_eye}")

        d = abs(right_eye[0] - left_eye[0])
        logging.debug(f"Pupillary Distance: {d}px")
        ar = self.width / self.height
        logging.debug(f"Aspect Ratio: {ar}")
        new_width = int(self.width * (STYLEGAN_PUPIL_DIST / d))
        new_height = int(new_width / ar)
        logging.debug(
            f"Scaling from (h x w): {self.height}x{self.width} to {new_height}x{new_width}"
        )
        scale = new_width / self.width
        logging.debug(f"Scale: {scale}x")
        img = cv2.resize(self.original_img, (new_width, new_height))

        crop_x = int((self.landmarks[96][0] * scale) - STYLEGAN_RIGHT_PUPIL[0])
        crop_y = int((self.landmarks[96][1] * scale) - STYLEGAN_RIGHT_PUPIL[1])
        logging.debug(f"Crop x,y: {crop_x}, {crop_y}")
        img, _, _ = safe_clip(
            img, crop_x, crop_y, STYLEGAN_WIDTH, STYLEGAN_WIDTH, FILL_COLOR
        )
        self.landmarks = scale_crop_points(self.landmarks, crop_x, crop_y, scale)
        logging.debug(f"Resulting image: {img.shape}")
        super().load_image(img)

    def get_pupils(self):
        return self.landmarks[LM_LEFT_PUPIL], self.landmarks[LM_RIGHT_PUPIL]
