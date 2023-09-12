import os
import sys
from facial_analysis.video import VideoToVideo
from facial_analysis.calc import AnalyzeFAI, AnalyzeOralCommissureExcursion, AnalyzeBrows, AnalyzeDentalArea, AnalyzeEyeArea

if len(sys.argv) != 2:
  print("Please call with: process_video [video file]")
  sys.exit(1)


filename = sys.argv[1]

base_filename = os.path.splitext(os.path.basename(filename))[0]
graph_filename = os.path.join(os.path.dirname(filename), base_filename + "-graph.png")
analyze_filename = os.path.join(os.path.dirname(filename), base_filename + "-analyze.mp4")
data_filename = os.path.join(os.path.dirname(filename), base_filename + "-data.csv")

v = VideoToVideo()
STATS = [AnalyzeFAI(), AnalyzeOralCommissureExcursion(), AnalyzeBrows(), AnalyzeDentalArea(), AnalyzeEyeArea()]
v.process(
  filename,
  analyze_filename,
  STATS
  )
v.plot_chart(graph_filename)
v.dump_data(data_filename)
# ,['oce.a','oce.b', 'eye.l', 'eye.r', 'd.eye']



