from facial_analysis.facial import load_face_image
from facial_analysis.calc import (
    AnalyzeFAI,
    AnalyzeOralCommissureExcursion,
    AnalyzeBrows,
    AnalyzeDentalArea,
    AnalyzeEyeArea,
)
from facial_analysis.video import VideoToVideo
import time
import argparse
import os
import logging
import torch


def process_image(input_file, output_file, points, crop):
    face = load_face_image(input_file, crop=crop)
    face.analyze()

    if points:
        face.draw_landmarks(numbers=True)
    face.save(output_file)
    print(f"Output file: {output_file}")


def process_video(input_file, output_base, points, crop):
    graph_filename = os.path.join(output_base + "-graph.png")
    analyze_filename = os.path.join(output_base + "-analyze.mp4")
    data_filename = os.path.join(output_base + "-data.csv")

    v = VideoToVideo(points, crop)
    STATS = [
        AnalyzeFAI(),
        AnalyzeOralCommissureExcursion(),
        AnalyzeBrows(),
        AnalyzeDentalArea(),
        AnalyzeEyeArea(),
    ]
    result = v.process(input_file, analyze_filename, STATS)

    if result:
        v.plot_chart(graph_filename)
        v.dump_data(data_filename)

        print(f"Video output: {analyze_filename}")
        print(f"Graph output: {graph_filename}")
        print(f"Data output: {data_filename}")
    else:
        print("Video analysis failed")


parser = argparse.ArgumentParser(description="Process an image.")
parser.add_argument(
    "--points",
    default=False,
    action="store_true",
    help="Display face landmarks points.",
)
parser.add_argument(
    "--crop", default=False, action="store_true", help="Crop/zoom to face."
)
parser.add_argument("input_file", type=str, help="Path to the input image file.")
parser.add_argument(
    "--device", type=str, default="detect", help="The GPU/CPU device to use."
)
parser.add_argument(
    "output_file",
    type=str,
    nargs="?",
    default=None,
    help="Path to the output image file.",
)

args = parser.parse_args()

device = args.device
if device == "detect":
    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "gpu" if torch.cuda.is_available() else "cpu"
    print("Detecting compute device")
print(f"Using device: {device}")

input_file = args.input_file
if args.output_file:
    print("You gave me an output file")
else:
    output_base, media_ext = os.path.splitext(input_file)
    output_file = output_base + "_output" + media_ext

print(f"Input file: {input_file}")
print(f"Media extension: {media_ext}")
print(f"Crop: {args.crop}")

start_time = time.time()
if media_ext.lower() == ".mp4":
    print("Video analysis")
    process_video(input_file, output_base, points=args.points, crop=args.crop)
else:
    print("Image analysis")
    process_image(
        input_file=input_file,
        output_file=output_file,
        points=args.points,
        crop=args.crop,
    )

end_time = time.time()

elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Elapsed Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
