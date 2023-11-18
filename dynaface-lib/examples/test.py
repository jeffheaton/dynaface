from facial_analysis.facial import load_face_image
from facial_analysis.calc import AnalyzeFAI, AnalyzeOralCommissureExcursion, AnalyzeBrows, AnalyzeDentalArea, AnalyzeEyeArea

import torch
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

#subject = "/content/drive/MyDrive/projects/tracy/samples/after.jpg"
subject = "/Users/jeff/data/facial/samples/2021-8-19.png"
#subject = "/content/drive/MyDrive/projects/tracy/samples/mark_ruffalo.jpg"
# 60 - 67

face = load_face_image(subject)
face.analyze()
face.draw_landmarks(numbers=True)
print(subject)
face.save("/Users/jeff/data/facial/samples/test.png")
#display_image(face.render_img, scale=1)
