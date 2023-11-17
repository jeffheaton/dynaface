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
  if crop: img = FindFace.crop(img)
  face.load_image(img)
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
    # bbox to spiga is x,y,w,h; however, facenet_pytorch deals in x1,y1,x2,y2. annoying
    bbox = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
    img_tmp = FindFace.crop(img)
    features = self.processor.inference(img, [bbox])

    # Prepare variables
    x0,y0,w,h = bbox
    landmarks2 = [(int(x[0]),int(x[1])) for x in np.array(features['landmarks'][0])]
    headpose = np.array(features['headpose'][0])
    return landmarks2, headpose

  def load_image(self, img):
    super().load_image(img)
    self.landmarks, self.headpose = self._find_landmarks(img)

    # 45 = left edge of left eye
    # 42 = right edge of left eye
    # 36 = right edge of right eye
    # 39 = left edge of right eye
    # 33 = nose (bottom center)

    self.right_eye = (int(self.landmarks[60][0] + self.landmarks[64][0]) // 2, \
      int(self.landmarks[60][1] + self.landmarks[64][1]) // 2)
    self.left_eye = (int(self.landmarks[68][0] + self.landmarks[72][0]) // 2, \
      int(self.landmarks[68][1] + self.landmarks[72][1]) // 2)

    self.nose = self.landmarks[57]
    self._estimate_mouth()
    self.pupillary_distance = abs(self.right_eye[0] - self.left_eye[0])
    self.pix2mm = 63/self.pupillary_distance

  def _estimate_mouth(self):
    x1 = int(1e50)
    y1 = int(1e50)
    x2 = 0
    y2 = 0
    for i in range(48,68):
      landmark = self.landmarks[i]
      x1 = min(x1,landmark[0])
      y1 = min(y1,landmark[1]-25)
      x2 = max(x2,landmark[0])
      y2 = max(y2,landmark[1]+25)

    self.mouth_p1 = (x1,y1)
    self.mouth_p2 = (x2,y2)

  def scan_vert(self, y):
    y+=self.nose[1]
    x1 = self.nose[0] - (self.mouth_width//2)
    x2 = self.nose[0] + (self.mouth_width//2)
    scan_line_color = self.original_hsv[y,x1:x2]
    scan_line = scan_line_color.mean(axis=1).astype(int)
    scan_line2 = np.append(scan_line[0] , scan_line[:-1])
    scan_diff = np.absolute(scan_line - scan_line2)
    white_line = np.repeat(255,self.mouth_width*3).reshape((self.mouth_width,3))
    teeth_pred = np.sqrt(((white_line - scan_line_color)**2).sum(axis=1)).astype(int)
    return scan_line_color #, teeth_pred

  def draw_landmarks(self, size=0.25, color=[0,255,255],numbers=False):
    for i,landmark in enumerate(self.landmarks):
      self.circle(landmark,radius=3,color=color)
      if numbers:
        self.write_text([landmark[0]+3,landmark[1]],str(i),size=0.5)
    self.circle(self.left_eye, color=color)
    self.circle(self.right_eye, color=color)


  def cluster_mouth_rect(self):
    x1, y1 = self.mouth_p1
    x2, y2 = self.mouth_p2

    if USE_HSV:
      mouth_img = self.original_hsv[y1:y2,x1:x2]
    else:
      mouth_img = self.original_img[y1:y2,x1:x2]

    mouth_pixels = mouth_img.reshape(mouth_img.shape[0]*mouth_img.shape[1],3)
    return KMeans(n_clusters=CLUST_NUM, random_state=42, n_init=1, init='k-means++').fit(mouth_pixels)


  def segment_line(self, kmeans, y):
    x1, y1 = self.mouth_p1
    x2, y2 = self.mouth_p2
    lst = []

    if USE_HSV:
      l = self.extract_horiz_hsv(y)
    else:
      l = self.extract_horiz(y)
    return kmeans.predict(l[x1:x2])

  def segment_display_line(self, kmeans, y):
    labels = self.segment_line(kmeans, y)

    lst = []
    for v in labels:
      lst.append(COLORS[v])
    l2 = np.array(lst)
    l2 = l2.reshape((1,l2.shape[0],l2.shape[1]))

    x1, y1 = self.mouth_p1
    x2, y2 = self.mouth_p2

    lfill = np.repeat(COLORS[-1].reshape((1,3)),x1,axis=0)
    lfill = lfill.reshape([1,lfill.shape[0],lfill.shape[1]])
    l2 = np.concatenate([lfill,l2],axis=1)

    l2 = np.repeat(l2,50,axis=0)
    return l2


  def stretch_scan_line(self, l):
    x1, y1 = self.mouth_p1
    x2, y2 = self.mouth_p2
    lst = []

    l2 = l.copy()
    l2[:x1,:] = 0
    l2[x2:,:] = 0
    l2 = l2.reshape((1,l2.shape[0],l2.shape[1]))
    l2 = np.repeat(l2,50,axis=0)
    return l2

  def scan_mouth(self, kmeans):
    x1, y1 = self.mouth_p1
    x2, y2 = self.mouth_p2

    if USE_HSV:
      mouth_img = self.original_hsv[y1:y2,x1:x2]
    else:
      mouth_img = self.original_img[y1:y2,x1:x2]

    for y in range(len(mouth_img)):
      labels = kmeans.predict(mouth_img[y])
      lst = []
      for v in labels:
        lst.append(COLORS[v])
      l2 = np.array(lst)
      l2 = l2.reshape((1,l2.shape[0],l2.shape[1]))
      mouth_img[y] = l2

    return mouth_img

  def mark_mouth(self, color=(255,0,0), thickness=3):
    cv2.rectangle(self.render_img, self.mouth_p1, self.mouth_p2, color, thickness)

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