import cv2
import numpy as np
from PIL import Image
import dlib
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import math
import json
import copy
from facial_analysis.util import PolyArea

# oral commissure
# medial canthus

COLORS = np.array([[255,255,255],[255,0,0],[0,255,0],[0,0,255],[0,255,255],[255,0,255],[255,255,0],[0,0,0]],dtype=np.uint8)
CLUST_NUM = 3
USE_HSV = False

def display_lowres(img, title=""):
  plt.imshow(img)
  plt.title(title)
  plt.show()

def display_image(data, scale):
  img = Image.fromarray(data, 'RGB')
  width, height = img.size

  img = img.resize((
        int(width * scale),
        int(height * scale)),
        Image.ANTIALIAS)
  display(img)

def load_image(filename):
  if not os.path.exists(filename):
    raise FileNotFoundError(os.path.abspath(filename))
  image = cv2.imread(filename)
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class ImageAnalysis:
  def load_image(self, img):
    self.original_img = img.copy()
    self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    self.render_img = img.copy()
    self.original_hsv = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2HSV).astype(np.int64)
    self.shape = self.original_img.shape

    self.text_font=cv2.FONT_HERSHEY_SIMPLEX
    self.text_size=0.75
    self.text_color=(255,255,255)
    self.text_thick=1
    self.text_back=5

    self.height, self.width = self.original_img.shape[:2]

    self.stats_right = 750

  def write_text(self, pos, txt,color=None, size = 1, thick = 1):
    size = self.text_size * size
    thick = int(self.text_thick * thick)
    if color is None: color = self.text_color
    cv2.putText(self.render_img, txt, pos, self.text_font,
                    size, (0,0,0), thick+self.text_back, cv2.LINE_AA)

    cv2.putText(self.render_img, txt, pos, self.text_font,
                    size, self.text_color, thick, cv2.LINE_AA)

  def write_text_sq(self, pos, txt,color=None):
    if color is None: color = self.text_color

    cv2.putText(self.render_img, txt, pos, self.text_font,
                    self.text_size, (0,0,0), self.text_thick+self.text_back, cv2.LINE_AA)
    w1 = cv2.getTextSize(txt, self.text_font, self.text_size, self.text_thick+self.text_back)[0][0]

    cv2.putText(self.render_img, txt, pos, self.text_font,
                    self.text_size, self.text_color, self.text_thick, cv2.LINE_AA)
    w2 = cv2.getTextSize(txt, self.text_font, self.text_size, self.text_thick)[0][0]



    cv2.putText(self.render_img, "2", (pos[0]+w1-5,pos[1]-5), self.text_font,
                    self.text_size, (0,0,0), self.text_thick+self.text_back, cv2.LINE_AA)

    cv2.putText(self.render_img, "2", (pos[0]+w1-5,pos[1]-5), self.text_font,
                    self.text_size, self.text_color, self.text_thick, cv2.LINE_AA)

  def hline(self, y, x1=None, x2=None, color=(0,255,255),width=5):
    if not x1: x1 = 0
    if not x2: x2 = self.render_img.shape[0]
    cv2.line(self.render_img, (x1,y), (x2,y), color, width)

  def vline(self, x, y1=None, y2=None, color=(0,255,255),width=5):
    if not y1: y1 = 0
    if not y2: y2 = self.render_img.shape[1]
    cv2.line(self.render_img, (x,y1), (x,y2), color, width)

  def line(self, pt1, pt2, color=(0,255,255),width=5):
    cv2.line(self.render_img, pt1, pt2, color, width)

  def circle(self, pt, color=(0,0,255),radius=None):
    if radius is None: radius = int(self.render_img.shape[0] // 200)
    cv2.circle(self.render_img, pt, radius, color, -1)

  def render_reset(self):
    self.render_img = self.original_img.copy()

  def extract_horiz(self, y, x1=None, x2=None):
    if not x1: x1 = 0
    if not x2: x2 = self.render_img.shape[0]
    return self.original_img[y,x1:x2]

  def extract_horiz_hsv(self, y, x1=None, x2=None):
    if not x1: x1 = 0
    if not x2: x2 = self.render_img.shape[0]
    return self.original_hsv[y,x1:x2]

  def save(self, filename):
    image = cv2.cvtColor(self.render_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename,image)

  def measure_polygon(self, contours, pix2mm, alpha=0.4, color=(0,0,255)):
    contours = np.array(contours)
    overlay = self.render_img.copy()
    cv2.fillPoly(overlay, pts = [contours], color =color)
    self.render_img = cv2.addWeighted(overlay, alpha, self.render_img, 1 - alpha, 0)
    contours = contours*pix2mm
    x = contours[:,0]
    y = contours[:,1]
    return PolyArea(x,y)

  def line(self, pt1, pt2, color=(255,0,0), thickness=3):
    cv2.line(self.render_img, pt1, pt2, color, thickness)

  # https://www.codeguru.com/multimedia/drawing-an-arrowline/
  def arrow_head(self, pt1, pt2, par=15):
    slopy = math.atan2( ( pt1[1] - pt2[1] ), ( pt1[0] - pt2[0] ) )
    cosy = math.cos( slopy )
    siny = math.sin( slopy )

    self.line(pt1, (pt1[0] + int( - par * cosy - ( par / 2.0 * siny ) ),
      pt1[1] + int( - par * siny + ( par / 2.0 * cosy ) ) ))
    self.line(pt1, (pt1[0] + int( - par * cosy + ( par / 2.0 * siny ) ),
      pt1[1] - int( par / 2.0 * cosy + par * siny ) ) )

  def arrow(self, pt1, pt2, color=(255,0,0), thickness=3, apt1=True, apt2=True):
    self.line(pt1, pt2, color, thickness)

    if apt1:
      self.arrow_head(pt1, pt2)

    if apt2:
      self.arrow_head(pt2, pt1)
