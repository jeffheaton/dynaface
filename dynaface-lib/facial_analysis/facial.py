import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from facial_analysis.image import load_image, ImageAnalysis
from facial_analysis.spiga.inference.config import ModelConfig
from facial_analysis.spiga.inference.framework import SPIGAFramework
from facial_analysis.util import PolyArea
from facial_analysis.calc import *
from facial_analysis.find_face import FindFace
import torch
import logging

STD_PUPIL_DIST = 63

LM_LEFT_PUPIL = 97
LM_RIGHT_PUPIL = 96

FILL_COLOR = [128, 128, 128]

STYLEGAN_WIDTH = 1024
STYLEGAN_LEFT_PUPIL = (640,480)
STYLEGAN_RIGHT_PUPIL = (380,480)
STYLEGAN_PUPIL_DIST = STYLEGAN_LEFT_PUPIL[0] - STYLEGAN_RIGHT_PUPIL[0]

SPIGA_MODEL = 'wflw'

STATS = [AnalyzeFAI(), AnalyzeOralCommissureExcursion(), AnalyzeBrows(), AnalyzeDentalArea(), AnalyzeEyeArea()]
_processor = None


def init_processor(device=None):
  global _processor

  if not device:
    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "gpu" if torch.cuda.is_available() else "cpu"
    device = "cpu"

  config = ModelConfig(dataset_name=SPIGA_MODEL, load_model_url=False)
  _processor = SPIGAFramework(config, device=device)


def load_face_image(filename, crop=True, stats=STATS, data_path=None):
  img = load_image(filename)
  face = AnalyzeFace(stats,data_path=data_path)
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

    # Ensure x and y are not negative
    x_offset = -min(x, 0)
    y_offset = -min(y, 0)
    x = max(x, 0)
    y = max(y, 0)

    # Ensure the region does not exceed the image boundaries
    x_end = min(x + width, cv2_image.shape[1])
    y_end = min(y + height, cv2_image.shape[0])
    clipped_region = cv2_image[y:y_end, x:x_end]

    # If the region is smaller than requested, fill the remaining area
    if clipped_region.shape[1] < width or clipped_region.shape[0] < height:
        new_image = np.full((height, width, 3), background, dtype=cv2_image.dtype)
        new_image[y_offset:y_offset + clipped_region.shape[0], x_offset:x_offset + clipped_region.shape[1]] = clipped_region
        clipped_region = new_image

    return clipped_region, x_offset, y_offset

def scale_crop_points(lst,crop_x,crop_y,scale):
  lst2 = []
  for pt in lst:
      lst2.append(
         (int(((pt[0]*scale)-crop_x)),
         int((pt[1]*scale)-crop_y)))
  return lst2

class AnalyzeFace (ImageAnalysis):
  
  def __init__(self, stats, data_path):
    global _processor

    if not _processor:
      init_processor()

    self.data_path = data_path
    self.left_eye = None
    self.right_eye = None
    self.nose = None
    self.calcs = stats
    self.headpose = [0,0,0]
    self.processor = _processor

  def get_all_stats(self):
    return [stat for obj in self.calcs for stat in obj.stats()]

  def _find_landmarks(self, img):

    bbox = FindFace.detect_face(img)

    if bbox is None: 
      bbox = [0,0,img.shape[1],img.shape[0]]
      logging.info("Could not detect face area")
    # bbox to spiga is x,y,w,h; however, facenet_pytorch deals in x1,y1,x2,y2. 
    bbox = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
    features = self.processor.inference(img, [bbox])

    # Prepare variables
    x0,y0,w,h = bbox
    landmarks2 = [(int(x[0]),int(x[1])) for x in np.array(features['landmarks'][0])]
    headpose = np.array(features['headpose'][0])
    return landmarks2, headpose

  def load_image(self, img, crop, eyes=None):
    super().load_image(img)
    self.landmarks, self._headpose = self._find_landmarks(img)
    self.pupillary_distance = abs(self.landmarks[LM_LEFT_PUPIL][0] - self.landmarks[LM_RIGHT_PUPIL][0])
    self.pix2mm = STD_PUPIL_DIST/self.pupillary_distance
    if crop:
      self.crop_stylegan(eyes)

  def draw_landmarks(self, size=0.25, color=[0,255,255],numbers=False):
    for i,landmark in enumerate(self.landmarks):
      self.circle(landmark,radius=3,color=color)
      if numbers:
        self.write_text([landmark[0]+3,landmark[1]],str(i),size=0.5)
    self.circle(self.left_eye, color=color)
    self.circle(self.right_eye, color=color)

  def measure(self, pt1, pt2, color=(255,0,0), thickness=3):
    self.arrow(pt1, pt2, color, thickness)
    d = math.dist(pt1,pt2) * self.pix2mm
    mp = [int((pt1[0]+pt2[0])//2),int((pt1[1]+pt2[1])//2)]
    self.write_text(mp,f"{d:.2f}mm")
    return d

  def analyze(self):
    result = {}
    for calc in self.calcs:
      result.update(calc.calc(self))
    return result
  
  def crop_stylegan(self, pupils=None):
    logging.info(f"Pupils provided: {pupils}")
    if pupils:
      left_eye, right_eye = pupils
    else:
      left_eye, right_eye = self.get_pupils()
      logging.info(f"Pupils calculated: l:{left_eye}, r:{right_eye}")

    d = abs(right_eye[0] - left_eye[0])
    logging.info(f"Pupillary Distance: {d}px")
    ar = self.width/self.height
    logging.info(f"Aspect Ratio: {ar}")
    new_width = int(self.width * (STYLEGAN_PUPIL_DIST/d))
    new_height = int(new_width / ar)
    logging.info(f"Scaling from (h x w): {self.height}x{self.width} to {new_height}x{new_width}")
    scale = new_width / self.width
    logging.info(f"Scale: {scale}x")
    img = cv2.resize(self.original_img, (new_width, new_height))

    crop_x = int((self.landmarks[96][0]*scale)-STYLEGAN_RIGHT_PUPIL[0])
    crop_y = int((self.landmarks[96][1]*scale)-STYLEGAN_RIGHT_PUPIL[1])
    logging.info(f"Crop x,y: {crop_x}, {crop_y}")
    img, _, _ = safe_clip(img, crop_x, crop_y, STYLEGAN_WIDTH, STYLEGAN_WIDTH, FILL_COLOR)
    self.landmarks = scale_crop_points(self.landmarks,crop_x,crop_y,scale)
    logging.info(f"Resulting image: {img.shape}")
    super().load_image(img)

  def get_pupils(self):
    return self.landmarks[LM_LEFT_PUPIL], self.landmarks[LM_RIGHT_PUPIL]

  