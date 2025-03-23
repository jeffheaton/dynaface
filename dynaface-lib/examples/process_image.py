import argparse
import os
from dynaface.facial import load_face_image
from dynaface.measures import AnalyzeFAI, AnalyzeOralCommissureExcursion
from dynaface import models

# Sample usage:
# python process_image.py --no-points https://data.heatonresearch.com/dynaface/sample/img4-1024-frontal.jpg output.jpg

parser = argparse.ArgumentParser(description="Process an image with facial analysis.")

# Boolean flag: show points by default, suppress with --no-points
parser.add_argument(
    "--no-points",
    action="store_false",
    dest="show_points",
    help="Disable display of facial landmarks.",
)
parser.set_defaults(show_points=True)

parser.add_argument("input_file", type=str, help="Path to the input image file.")
parser.add_argument(
    "output_file",
    type=str,
    nargs="?",
    default=None,
    help="Path to the output image file (optional).",
)

args = parser.parse_args()
output_path = args.output_file or os.path.splitext(args.input_file)[0] + "_output.jpg"

device = models.detect_device()
print(f"Detected device: {device}")
path = models.download_models()
models.init_models(path, device)

face = load_face_image(
    args.input_file, measures=[AnalyzeFAI(), AnalyzeOralCommissureExcursion()]
)
face.analyze()

if args.show_points:
    face.draw_landmarks(numbers=True)

face.save(output_path)
print(f"Saved annotated image to {output_path}")
