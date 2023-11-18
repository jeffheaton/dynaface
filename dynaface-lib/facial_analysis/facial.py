import os
import cv2
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import urllib
import bz2
from facenet_pytorch import MTCNN
from facial_analysis.image import load_image, ImageAnalysis
from facial_analysis.spiga.inference.config import ModelConfig
from facial_analysis.spiga.inference.framework import SPIGAFramework
from facial_analysis.util import PolyArea
from facial_analysis.calc import *
import torch

STD_PUPIL_DIST = 63

LM_LEFT_PUPIL = 97
LM_RIGHT_PUPIL = 96

STYLEGAN_WIDTH = 1024
STYLEGAN_LEFT_PUPIL = (640,480)
STYLEGAN_RIGHT_PUPIL = (380,480)
STYLEGAN_PUPIL_DIST = STYLEGAN_LEFT_PUPIL[0] - STYLEGAN_RIGHT_PUPIL[0]

STATS = [AnalyzeFAI(), AnalyzeOralCommissureExcursion(), AnalyzeBrows(), AnalyzeDentalArea(), AnalyzeEyeArea()]
_processor = None


def init_processor(device=None):
  global _processor

  if not device:
    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "gpu" if torch.cuda.is_available() else "cpu"
    device = "cpu"
  _processor = SPIGAFramework(ModelConfig('wflw'), device=device)


def load_face_image(filename, crop=True, stats=STATS):
  img = load_image(filename)
  face = AnalyzeFace(stats)
  face.load_image(img, crop)
  return face

def find_facial_path():
    home_directory = os.path.expanduser( '~' )
    path = os.path.join( home_directory, '.facial_analysis' )
    try:
        os.makedirs(path)
    except FileExistsError:
       # directory already exists
       pass
    return path

def add_gray_border_to_min_dimension(img, min_dimension):
    # Get current dimensions of the image
    height, width = img.shape[:2]

    # Calculate padding needed for width and height
    if width < min_dimension:
        left_padding = (min_dimension - width) // 2
        right_padding = min_dimension - width - left_padding
    else:
        left_padding = right_padding = 0

    if height < min_dimension:
        top_padding = (min_dimension - height) // 2
        bottom_padding = min_dimension - height - top_padding
    else:
        top_padding = bottom_padding = 0

    # Add gray border around the image
    gray_color = [128, 128, 128]  # RGB value for gray
    bordered_img = cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=gray_color)

    return bordered_img

def scale_crop_points(lst,crop_x,crop_y,scale):
  lst2 = []
  for pt in lst:
      lst2.append(
         (int(((pt[0]*scale)-crop_x)),
         int((pt[1]*scale)-crop_y)))
  return lst2

class FindFace():
  mtcnn = MTCNN(keep_all=True, device="cpu")

  def detect_face(img):
    boxes, _ = FindFace.mtcnn.detect(img)
    if boxes is None: return None
    return boxes[0]
    
  def crop(img):
    boxes, _ = FindFace.mtcnn.detect(img)
    if boxes is not None:
        # Assuming the first face detected
        box = boxes[0]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        return img[y1:y2, x1:x2]
    
    return None

class AnalyzeFace (ImageAnalysis):
  
  def __init__(self, stats):
    global _processor

    if not _processor:
      init_processor()

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
    if bbox is None: bbox = [0,0,img.shape[1],img.shape[0]]
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
    if pupils:
      left_eye, right_eye = pupils
    else:
      left_eye, right_eye = self.get_pupils()

    d = abs(right_eye[0] - left_eye[0])
    ar = self.width/self.height
    new_width = int(self.width * (STYLEGAN_PUPIL_DIST/d))
    new_height = int(new_width / ar)
    scale = new_width / self.width
    img = cv2.resize(self.original_img, (new_width, new_height))
    img = add_gray_border_to_min_dimension(img,STYLEGAN_WIDTH)
    crop_x = int((self.landmarks[96][0]*scale)-STYLEGAN_RIGHT_PUPIL[0])
    crop_y = int((self.landmarks[96][1]*scale)-STYLEGAN_RIGHT_PUPIL[1])
    self.landmarks = scale_crop_points(self.landmarks,crop_x,crop_y,scale)
    img = img[crop_y:crop_y+STYLEGAN_WIDTH,crop_x:crop_x+STYLEGAN_WIDTH]
    super().load_image(img)

  def get_pupils(self):
    return self.landmarks[LM_LEFT_PUPIL], self.landmarks[LM_RIGHT_PUPIL]

  
