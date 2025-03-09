import copy
import logging
import math
import time

import cv2
import numpy as np
import torch
from facial_analysis import measures, models, util
from facial_analysis.image import ImageAnalysis, load_image
from facial_analysis.spiga.inference.config import ModelConfig
from facial_analysis.spiga.inference.framework import SPIGAFramework
from facial_analysis.lateral import analyze_lateral

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


def load_face_image(
    filename,
    crop=True,
    stats=None,
    tilt_threshold=DEFAULT_TILT_THRESHOLD,
):
    if stats is None:
        stats = measures.all_measures()
    img = load_image(filename)
    face = AnalyzeFace(stats, tilt_threshold=tilt_threshold)
    face.load_image(img, crop)
    return face


class AnalyzeFace(ImageAnalysis):
    pd = STD_PUPIL_DIST

    def __init__(self, measures=None, tilt_threshold=DEFAULT_TILT_THRESHOLD):
        self.original_img = None
        self.left_eye = None
        self.right_eye = None
        self.nose = None
        self._headpose = None
        if measures is None:
            self.measures = measures.all_measures()
        else:
            self.measures = measures
        self.headpose = [0, 0, 0]
        self.landmarks = []
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

    def is_lateral(self):
        if self._headpose is None:
            return False
        yaw, pitch, roll = self._headpose[:3]
        logger.debug(f"Headpose: yaw:{yaw}, pitch:{pitch}, roll:{roll}")
        return abs(yaw) > 15

    def _overlay_lateral_analysis(self, c):
        """Scales and overlays the lateral analysis image onto self.render_img at the top-right."""
        if c is None:
            return

        # Scale 'c' to a height of 1024 while maintaining aspect ratio
        c_height, c_width = c.shape[:2]
        scale_factor = 1024 / c_height
        print(f"Height x width: {c_height} {c_width}")
        print(f"Scale Factor: {scale_factor}")

        new_width = int(c_width * scale_factor)
        c_resized = cv2.resize(c, (new_width, 1024))
        print(f"New width (before limiting to MAX_INSERT_WIDTH): {new_width}")

        MAX_INSERT_WIDTH = 1024  # Disabled currently, as 1024 is max

        # Limit width to a maximum of MAX_INSERT_WIDTH
        if new_width > MAX_INSERT_WIDTH:
            new_width = MAX_INSERT_WIDTH  # Crop to MAX_INSERT_WIDTH max width
            c_resized = c_resized[:, :MAX_INSERT_WIDTH]  # Keep the left side

        print(f"Final new width: {new_width}")

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

        if self.is_lateral() and self.landmarks:
            self.crop_lateral()
        elif self.landmarks is not None:
            logger.debug("Landmarks located")
            self.calc_pd()

            if crop:
                logger.debug("Cropping")
                self.crop_stylegan(pupils=pupils)
        else:
            logger.info("No face detected")
            return False

        if self.is_lateral():
            ## MODIFY this part
            p = util.cv2_to_pil(self.render_img)
            c = analyze_lateral(p)
            c = util.trim_sides(c)
            cv2.imwrite("debug_overlay.png", c)
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

    def calculate_face_rotation(self):
        p = util.get_pupils(self.landmarks)
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

        print("Before expansion:", crop_x, crop_y, w, h)

        # Compute vertical expansion amounts
        expand_top = int(h * INFLATE_LATERAL_TOP)
        expand_bottom = int(h * INFLATE_LATERAL_BOTTOM)

        # Ensure new crop_y does not go out of bounds
        crop_y = max(0, crop_y - expand_top)  # Move up by expand_top

        # Adjust height
        h = h + expand_top + expand_bottom

        print("After expansion:", crop_x, crop_y, w, h)

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
        self.orig_pupils = util.get_pupils(self.landmarks)

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
            pupils = util.get_pupils(landmarks=self.landmarks)

        d, _ = util.calc_pd(pupils)

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
        self.pupillary_distance, self.pix2mm = util.calc_pd(self.get_pupils())

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

        try:
            self.face_rotation = obj[5]
        except IndexError:
            self.face_rotation = 0
        except TypeError:
            self.face_rotation = 0

    def find_pupils(self):
        return util.get_pupils(self.landmarks)

    def calc_bisect(self):
        return util.bisecting_line_coordinates(img_size=1024, pupils=self.find_pupils())
