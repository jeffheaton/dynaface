import numpy as np

class AnalyzeFAI:
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

class AnalyzeBrows:
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

    
    return {'brow_diff':diff}

class AnalyzeDentalArea():
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

    eye_area_diff = abs(right_eye_area-left_eye_area)

    face.write_text_sq((face.landmarks[66][0]-50,face.landmarks[66][1]+20),f"R={round(right_eye_area,2)}mm")
    face.write_text_sq((face.landmarks[74][0]-50,face.landmarks[74][1]+20),f"L={round(left_eye_area,2)}mm")
    face.write_text_sq((face.stats_right,face.landmarks[74][1]+50),f"d.eyes={round(eye_area_diff,2)}mm")

    return {'left_eye_area':left_eye_area, 'right_eye_area':right_eye_area, 'eye_area_diff':eye_area_diff}