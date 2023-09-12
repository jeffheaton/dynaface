from facial_analysis.facial import load_face_image
from facial_analysis.calc import AnalyzeFAI, AnalyzeOralCommissureExcursion, AnalyzeBrows, AnalyzeDentalArea, AnalyzeEyeArea

import torch

import argparse

parser = argparse.ArgumentParser(description="Process an image.")
    
# Define the -points option. You can specify its type, in this case, int.
parser.add_argument("-points", type=int, default=0, help="Number of points for the operation.")

# Define positional arguments for the input and output files.
parser.add_argument("input_file", type=str, help="Path to the input image file.")
parser.add_argument("output_file", type=str, nargs='?', default=None, help="Path to the output image file.")

args = parser.parse_args()
print(args)

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
