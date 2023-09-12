from facial_analysis.facial import load_face_image
from facial_analysis.calc import AnalyzeFAI, AnalyzeOralCommissureExcursion, AnalyzeBrows, AnalyzeDentalArea, AnalyzeEyeArea
from facial_analysis.video import VideoToVideo
import torch
import argparse
import os

def process_image(input_file, output_file, points):
    face = load_face_image(input_file)
    face.analyze()

    if points:
        face.draw_landmarks(numbers=True)
    face.save(output_file)
    print(f"Output file: {output_file}")

def process_video(input_file, output_base):
    graph_filename = os.path.join(output_base + "-graph.png")
    analyze_filename = os.path.join(output_base + "-analyze.mp4")
    data_filename = os.path.join(output_base + "-data.csv")

    v = VideoToVideo()
    STATS = [AnalyzeFAI(), AnalyzeOralCommissureExcursion(), AnalyzeBrows(), AnalyzeDentalArea(), AnalyzeEyeArea()]
    result = v.process(
        input_file,
        analyze_filename,
        STATS)
    
    if result:
        v.plot_chart(graph_filename)
        v.dump_data(data_filename)

        print(f"Video output: {analyze_filename}")
        print(f"Graph output: {graph_filename}")
        print(f"Data output: {data_filename}")
    else:
        print("Video analysis failed")


parser = argparse.ArgumentParser(description="Process an image.")
parser.add_argument("--points", default=False, action="store_true", help="Display face landmarks points.")
parser.add_argument("input_file", type=str, help="Path to the input image file.")
parser.add_argument("output_file", type=str, nargs='?', default=None, help="Path to the output image file.")

args = parser.parse_args()
print(args)

input_file = args.input_file
if args.output_file:
    print("You gave me an output file")
else:
    output_base, media_ext = os.path.splitext(input_file)
    output_file = output_base + "_output" + media_ext

print(f"Input file: {input_file}")
print(f"Media extension: {media_ext}")
if media_ext.lower()=='.mp4':
    print("Video analysis")
    process_video(input_file, output_base)
else:
    print("Image analysis")
    process_image(input_file=input_file,output_file=output_file,points=args.points)



# python ./examples/process_media.py /Users/jeff/data/facial/samples/2021-8-19.png