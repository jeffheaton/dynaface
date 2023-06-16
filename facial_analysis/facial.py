import cv2
import numpy as np
from PIL import Image
import dlib
from matplotlib import pyplot as plt
from image_analysis.image import load_image
  
  def load_face_image(filename, crop=True):
    img = load_image(filename)
    face = AnalyzeFace()
    face.load_image(image)
    if crop: face.crop_stylegan()
    return face

class StyleGANCrop:
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('/content/shape_predictor_5_face_landmarks.dat')

  def find_eyes(self, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = StyleGANCrop.detector(gray, 0)

    if len(rects) == 0:
      raise ValueError("No faces detected")
    elif len(rects) > 1:
      raise ValueError("Multiple faces detected")

    shape = StyleGANCrop.predictor(gray, rects[0])
    features = []

    for i in range(0, 5):
      features.append((i, (shape.part(i).x, shape.part(i).y)))

    return (int(features[3][1][0] + features[2][1][0]) // 2, \
      int(features[3][1][1] + features[2][1][1]) // 2), \
      (int(features[1][1][0] + features[0][1][0]) // 2, \
      int(features[1][1][1] + features[0][1][1]) // 2)


  def measure_stylegan(self, img):
    left_eye, right_eye = self.find_eyes(img)
    d = abs(right_eye[0] - left_eye[0])
    z = 255/d
    ar = img.shape[0]/img.shape[1]
    w = img.shape[1] * z
    img2 = cv2.resize(img, (int(w), int(w*ar)))
    bordersize = 1024
    img3 = cv2.copyMakeBorder(
        img2,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,value=(0,0,0))

    left_eye2, right_eye2 = self.find_eyes(img3)

    crop1 = left_eye2[0] - 385
    crop0 = left_eye2[1] - 490

    return [(int(w), int(w*ar)), crop0, crop1]

  def size_crop(self, img, sz, crop0, crop1):
    img2 = cv2.resize(img, sz)
    bordersize = 1024
    img3 = cv2.copyMakeBorder(
        img2,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
    return img3[crop0:crop0+1024,crop1:crop1+1024]


  def crop_stylegan(self, img):
    left_eye, right_eye = self.find_eyes(img)
    d = abs(right_eye[0] - left_eye[0])
    z = 255/d
    ar = img.shape[0]/img.shape[1]
    w = img.shape[1] * z
    img2 = cv2.resize(img, (int(w), int(w*ar)))
    bordersize = 1024
    img3 = cv2.copyMakeBorder(
        img2,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,value=(0,0,0))

    left_eye2, right_eye2 = self.find_eyes(img3)

    crop1 = left_eye2[0] - 385
    crop0 = left_eye2[1] - 490
    return img3[crop0:crop0+1024,crop1:crop1+1024]

class AnalyzeFace (ImageAnalysis):
  crop = StyleGANCrop()
  processor = SPIGAFramework(ModelConfig('wflw'))

  def __init__(self):
    self.left_eye = None
    self.right_eye = None
    self.nose = None

  def _find_landmarks(self, img):

    bbox = [0,0,img.shape[1],img.shape[0]]
    features = AnalyzeFace.processor.inference(img, [bbox])

    # Prepare variables
    x0,y0,w,h = bbox
    landmarks2 = [(int(x[0]),int(x[1])) for x in np.array(features['landmarks'][0])]
    return landmarks2

    #landmarks = []
    #for i in range(68):
    #  landmarks.append([int(shape.part(i).x),int(shape.part(i).y)])
    #return landmarks

  def load_image(self, img):
    super().load_image(img)
    self.landmarks = self._find_landmarks(img)

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

  def crop_stylegan(self):
    img = self.crop.crop_stylegan(self.original_img)
    self.load_image(img)

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

  def mark_teeth(self, y, teeth_pred):
    x1 = self.nose[0] - (self.mouth_width//2)
    y+=self.nose[1]
    for i in range(len(teeth_pred)):
      if teeth_pred[i]<150:
        self.render_img[y,x1+i] = (0,0,255)

  def draw_landmarks(self, size=0.25, color=[0,255,255],numbers=False):
    for i,landmark in enumerate(self.landmarks):
      self.circle(landmark,radius=2,color=color)
      if numbers:
        self.write_text(landmark,str(i))
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

  def dental_area_initial(self):
      # Mark inner mouth area as found by neural network
      z = np.zeros( self.original_img.shape, dtype=np.uint8)
      contours = [self.landmarks[88],
                  self.landmarks[89],
                  self.landmarks[90],
                  self.landmarks[91],
                  self.landmarks[92],
                  self.landmarks[93],
                  self.landmarks[94],
                  self.landmarks[95]]

      contours = np.array(contours)
      cv2.fillPoly(z, pts = [contours], color =(255,255,255))

      # Extract vertical mouth scan lines
      dental = []
      for x in range(self.original_img.shape[0]):
        s = z[:,x].sum()
        if s:
          s = np.where(z[:,x,0])
          dental.append( {'x':x, 'y1': min(s[0]), 'y2':max(s[0])} )
      return dental

  def measure_dental_area(self):
    contours = [self.landmarks[88],
              self.landmarks[89],
              self.landmarks[90],
              self.landmarks[91],
              self.landmarks[92],
              self.landmarks[93],
              self.landmarks[94],
              self.landmarks[95]]

    contours = np.array(contours)
    contours = contours*self.pix2mm
    x = contours[:,0]
    y = contours[:,1]
    return PolyArea(x,y)

  def measure_eye_area(self):
    right_eye_area = self.measure_polygon(
      [self.landmarks[60],
      self.landmarks[61],
      self.landmarks[62],
      self.landmarks[63],
      self.landmarks[64],
      self.landmarks[65],
      self.landmarks[66],
      self.landmarks[67]])

    left_eye_area = self.measure_polygon(
      [self.landmarks[68],
      self.landmarks[69],
      self.landmarks[70],
      self.landmarks[71],
      self.landmarks[72],
      self.landmarks[73],
      self.landmarks[74],
      self.landmarks[75]])

    eye_area_diff = abs(right_eye_area-left_eye_area)
    #66,74

    self.write_text_sq((self.landmarks[66][0]-50,self.landmarks[66][1]+20),f"R={round(right_eye_area,2)}mm")
    self.write_text_sq((self.landmarks[74][0]-50,self.landmarks[74][1]+20),f"L={round(left_eye_area,2)}mm")
    self.write_text_sq((self.stats_right,self.landmarks[74][1]+50),f"d.eyes={round(eye_area_diff,2)}mm")
    return left_eye_area, right_eye_area, eye_area_diff

    #self.write_text((10, 950),f"L={round(left_eye_area,2)}mm^2, R={round(right_eye_area,2)}mm^2, dif={round(eye_area_diff)}")
    #self.write_text((10, 950),f"Eyes: L={round(left_eye_area,2)}mm^2, R={round(right_eye_area,2)}mm^2, dif={round(eye_area_diff)}")
    #self.write_text((10, 950),f"Eyes: L={round(left_eye_area,2)}mm^2, R={round(right_eye_area,2)}mm^2, dif={round(eye_area_diff)}")

  def measure_brows(self):
    # left brow
    contours = [self.landmarks[34],
              self.landmarks[35],
              self.landmarks[36],
              self.landmarks[37]]

    contours = np.array(contours)
    x = contours[:,0]
    y = contours[:,1]
    left_brow_idx = np.argmin(y)
    left_brow_y = y[left_brow_idx]
    left_brow_x = x[left_brow_idx]
    self.arrow((left_brow_x, left_brow_y), (1024,left_brow_y),apt2=False)

    # right brow
    contours = [self.landmarks[42],
              self.landmarks[43],
              self.landmarks[44],
              self.landmarks[45]]

    contours = np.array(contours)
    x = contours[:,0]
    y = contours[:,1]
    right_brow_idx = np.argmin(y)
    right_brow_y = y[right_brow_idx]
    right_brow_x = x[right_brow_idx]
    self.arrow((right_brow_x, right_brow_y), (1024,right_brow_y),apt2=False)

    diff = abs(left_brow_y-right_brow_y)*self.pix2mm
    self.write_text((self.stats_right, min(left_brow_y,right_brow_y)-10),f"d.brow={diff:.2f} mm")
    return diff


  def analyze(self):
    d1 = self.measure(self.landmarks[64],self.landmarks[76])
    d2 = self.measure(self.landmarks[68],self.landmarks[82])
    if d1>d2:
      fai = d1 - d2
    else:
      fai = d2 - d1

    dental_sqmm = self.measure_dental_area()

    fai_pt = (self.stats_right,self.landmarks[82][1])
    self.write_text(fai_pt, f"FAI={fai:.2f} mm")
    dental_pt = (self.stats_right,self.landmarks[85][1]+20)
    self.write_text_sq(dental_pt,f"dental={round(dental_sqmm,2)}mm")
    dental_area = self.dental_area_initial()

    for scan in dental_area:
      x, y1, y2 = scan['x'], scan['y1'], scan['y2']
      for y in range(y1,y2):
        self.render_img[y][x][2] = 255

    left_eye_area, right_eye_area, eye_area_diff = self.measure_eye_area()
    brow_diff = self.measure_brows()
    return {'fai': fai, 'left_eye_area':left_eye_area, 'right_eye_area':right_eye_area, 'eye_area_diff':eye_area_diff, 'brow_diff':brow_diff}
