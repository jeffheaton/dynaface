import copy
import logging
import math
import time

import cv2
import numpy as np
import torch
from facial_analysis import measures, models
from facial_analysis.image import ImageAnalysis, load_image
from facial_analysis.spiga.inference.config import ModelConfig
from facial_analysis.spiga.inference.framework import SPIGAFramework
from facial_analysis import util

STYLEGAN_WIDTH = 1024
STYLEGAN_LEFT_PUPIL = (640, 480)
STYLEGAN_RIGHT_PUPIL = (380, 480)
STYLEGAN_PUPIL_DIST = STYLEGAN_LEFT_PUPIL[0] - STYLEGAN_RIGHT_PUPIL[0]

STD_PUPIL_DIST = 63
DEFAULT_TILT_THRESHOLD = 3

LM_LEFT_PUPIL = 97
LM_RIGHT_PUPIL = 96

FILL_COLOR = [128, 128, 128]

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


def load_face_image(filename, crop=True, stats=None):
    if stats is None:
        stats = measures.all_measures()
    img = load_image(filename)
    face = AnalyzeFace(stats)
    face.load_image(img, crop)
    return face


class AnalyzeFace(ImageAnalysis):
    pd = STD_PUPIL_DIST

    def __init__(self, measures=None):
        self.original_img = None
        self.left_eye = None
        self.right_eye = None
        self.nose = None
        if measures is None:
            self.measures = measures.all_measures()
        else:
            self.measures = measures
        self.headpose = [0, 0, 0]
        self.landmarks = []
        self.pupillary_distance = 0
        self.tilt_threshold = DEFAULT_TILT_THRESHOLD
        self.pix2mm = 1
        self.face_rotation = None

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
            logging.info("Could not detect face area")
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

    def load_image(self, img, crop, eyes=None):
        super().load_image(img)
        logger.debug("Low level-image loaded")
        self.landmarks, self._headpose = self._find_landmarks(img)

        if self.landmarks is not None:
            logger.debug("Landmarks located")
            self.calc_pd()

            if crop:
                logger.debug("Cropping")
                self.crop_stylegan(eyes)
        else:
            logger.info("No face detected")

    def draw_landmarks(self, size=0.25, color=[0, 255, 255], numbers=False):
        if self.landmarks is None:
            return
        for i, landmark in enumerate(self.landmarks):
            self.circle(landmark, radius=3, color=color)
            if numbers:
                self.write_text([landmark[0] + 3, landmark[1]], str(i), size=0.5)
        self.circle(self.left_eye, color=color)
        self.circle(self.right_eye, color=color)

    def measure(self, pt1, pt2, color=(255, 0, 0), thickness=3, render=True, dir="r"):
        if render:
            self.arrow(pt1, pt2, color, thickness)
        d = math.dist(pt1, pt2) * self.pix2mm
        txt = f"{d:.2f}mm"
        m = self.calc_text_size(txt)
        if dir == "r":
            mp = [int((pt1[0] + pt2[0]) // 2) + 15, int((pt1[1] + pt2[1]) // 2)]
        else:
            mp = [
                int((pt1[0] + pt2[0]) // 2) - (m[0][0] + 15),
                int((pt1[1] + pt2[1]) // 2),
            ]
        if render:
            self.write_text(mp, txt)
        return d

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

    def crop_stylegan(self, pupils=None):
        pupils = util.get_pupils(self.landmarks) if not pupils else pupils

        # Rotate, if needed
        if pupils:
            self.face_rotation = util.calculate_face_rotation(pupils)
            tilt = measures.to_degrees(self.face_rotation)
            if abs(self.tilt_threshold) > self.tilt_threshold:
                img2 = util.straighten(self.original_img, pupils)
                self.landmarks, self._headpose = self._find_landmarks(img2)
            else:
                img2 = self.original_img
        else:
            img2 = self.original_img
            self.face_rotation = 0

        # Crop, if needed
        img2, landmarks2 = util.crop_stylegan(
            img=img2, pupils=pupils, landmarks=self.landmarks
        )

        self.landmarks = landmarks2

        # Reload Image
        super().load_image(img2)

    def calc_pd(self):
        self.pupillary_distance, self.pix2mm = util.calc_pd(self.landmarks)

    def get_pupils(self):
        return util.get_pupils(self.landmarks)

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
        self.face_rotation = obj[4] if len(obj) > 4 else 0
