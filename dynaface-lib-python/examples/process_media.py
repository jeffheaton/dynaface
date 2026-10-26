import argparse
import os
import time

from dynaface.facial import load_face_image
from dynaface.measures import (
    AnalyzeBrows,
    AnalyzeDentalArea,
    AnalyzeEyeArea,
    AnalyzeFAI,
    AnalyzeOralCommissureExcursion,
)
from video import VideoToVideo
from dynaface import models

# Sample usage, from dynaface-lib-python:
# python examples/process_media.py --crop tests_data/img2-1024-right-lateral.jpg lateral.jpg
# python examples/process_media.py tests_data/video1-512.mp4 video.mp4


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


parser = argparse.ArgumentParser(description="Process an image or an .mp4 video.")
parser.add_argument(
    "--points",
    default=False,
    action="store_true",
    help="Display face landmarks points.",
)
parser.add_argument(
    "--crop", default=False, action="store_true", help="Crop/zoom to face."
)
parser.add_argument("input_file", type=str, help="Path to the input image or video.")
parser.add_argument(
    "--device", type=str, default="detect", help="The GPU/CPU device to use."
)
parser.add_argument(
    "output_file",
    type=str,
    nargs="?",
    default=None,
    help="Path to the output image, or the base name for video outputs.",
)

# process arguments
args = parser.parse_args()

# Detect device and download models
device = models.detect_device() if args.device == "detect" else args.device
print(f"Detected device: {device}")
path = models.download_models()
models.init_models(path, device)

input_file = args.input_file
media_ext = os.path.splitext(input_file)[1]
if args.output_file:
    output_file = args.output_file
    output_base = os.path.splitext(output_file)[0]
else:
    output_base = os.path.splitext(input_file)[0]
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
