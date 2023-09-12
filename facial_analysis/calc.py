import numpy as np

class AnalyzeFAI:
  def stats(self):
    return ['fai']
  def calc(self, face, render=True):
    d1 = face.measure(face.landmarks[64],face.landmarks[76])
    d2 = face.measure(face.landmarks[68],face.landmarks[82])
    if d1>d2:
      fai = d1 - d2
    else:
      fai = d2 - d1

    fai_pt = (face.stats_right,face.landmarks[82][1])
    if render:
      face.write_text(fai_pt, f"FAI={fai:.2f} mm")
    return {'fai': fai}

class AnalyzeOralCommissureExcursion:
  def stats(self):
    return ['oce.a','oce.b']
  def calc(self, face, render=True):
    oce_a = face.measure(face.landmarks[76],face.landmarks[85])
    oce_b = face.measure(face.landmarks[82],face.landmarks[85])
    return {'oce.a': oce_a, 'oce.b': oce_b}

class AnalyzeBrows:
  def stats(self):
    return ['brow.d']

  def calc(self, face, render=True):
    # left brow
    contours = [face.landmarks[34],
              face.landmarks[35],
              face.landmarks[36],
              face.landmarks[37]]

    contours = np.array(contours)
    x = contours[:,0]
    y = contours[:,1]
    left_brow_idx = np.argmin(y)
    left_brow_y = y[left_brow_idx]
    left_brow_x = x[left_brow_idx]
    if render:
      face.arrow((left_brow_x, left_brow_y), (1024,left_brow_y),apt2=False)

    # right brow
    contours = [face.landmarks[42],
              face.landmarks[43],
              face.landmarks[44],
              face.landmarks[45]]

    contours = np.array(contours)
    x = contours[:,0]
    y = contours[:,1]
    right_brow_idx = np.argmin(y)
    right_brow_y = y[right_brow_idx]
    right_brow_x = x[right_brow_idx]

    # Diff
    diff = abs(left_brow_y-right_brow_y)*face.pix2mm
    if render:
      face.arrow((right_brow_x, right_brow_y), (1024,right_brow_y),apt2=False)
      face.write_text((face.stats_right, min(left_brow_y,right_brow_y)-10),f"d.brow={diff:.2f} mm")
    
    return {'brow.d':diff}

class AnalyzeDentalArea():
  def stats(self):
    return ['dental_area']
  def calc(self, face, render=True):
    contours = [face.landmarks[88],
              face.landmarks[89],
              face.landmarks[90],
              face.landmarks[91],
              face.landmarks[92],
              face.landmarks[93],
              face.landmarks[94],
              face.landmarks[95]]

    contours = np.array(contours) #contours = contours*face.pix2mm
    dental_area = face.measure_polygon(contours, face.pix2mm)
    dental_pt = (face.stats_right,face.landmarks[85][1]+20)
    face.write_text_sq(dental_pt,f"dental={round(dental_area,2)}mm")
    return {'dental_area': dental_area}

class AnalyzeEyeArea():
  def stats(self):
    return ['eye.l', 'eye.r', 'eye.d', 'eye.rlr', 'eye.rrl']
  def calc(self, face, render=True):
    right_eye_area = face.measure_polygon(
      [face.landmarks[60],
      face.landmarks[61],
      face.landmarks[62],
      face.landmarks[63],
      face.landmarks[64],
      face.landmarks[65],
      face.landmarks[66],
      face.landmarks[67]], face.pix2mm)

    left_eye_area = face.measure_polygon(
      [face.landmarks[68],
      face.landmarks[69],
      face.landmarks[70],
      face.landmarks[71],
      face.landmarks[72],
      face.landmarks[73],
      face.landmarks[74],
      face.landmarks[75]], face.pix2mm)

    eye_area_diff = round(abs(right_eye_area-left_eye_area),2)
    eye_ratio_lr = round(left_eye_area/right_eye_area,2)
    eye_ratio_rl = round(right_eye_area/left_eye_area,2)

    face.write_text_sq((face.landmarks[66][0]-50,face.landmarks[66][1]+20),f"R={round(right_eye_area,2)}mm")
    face.write_text_sq((face.landmarks[74][0]-50,face.landmarks[74][1]+20),f"L={round(left_eye_area,2)}mm")
    face.write_text_sq((face.stats_right,face.landmarks[74][1]+50),f"d.eye={round(eye_area_diff,2)}mm")
    face.write_text_sq((face.stats_right,face.landmarks[74][1]+80),f"rlr.eye={round(eye_ratio_lr,2)}mm")
    face.write_text_sq((face.stats_right,face.landmarks[74][1]+110),f"rrl.eye={round(eye_ratio_rl,2)}mm")

    return {'eye.l':left_eye_area, 'eye.r':right_eye_area, 'eye.d':eye_area_diff, 'eye.rlr':eye_ratio_lr, 'eye.rrl':eye_ratio_rl}