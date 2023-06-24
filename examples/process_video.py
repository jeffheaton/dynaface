import os
import sys
from facial_analysis.video import VideoToVideo
from facial_analysis.calc import OralCommissureExcursion, AnalyzeEyeArea

if len(sys.argv) != 2:
  print("Please call with: process_vide [video file]")
  sys.exit(1)


filename = sys.argv[1]

base_filename = os.path.splitext(os.path.basename(filename))[0]
graph_filename = os.path.join(os.path.dirname(filename), base_filename + "-graph.png")
analyze_filename = os.path.join(os.path.dirname(filename), base_filename + "-analyze.mp4")

v = VideoToVideo()
STATS = [OralCommissureExcursion(),AnalyzeEyeArea()]
v.process(
  filename,
  analyze_filename,
  STATS
  )
v.plot_chart(graph_filename,['oce.a','oce.b', 'eye.l', 'eye.r', 'eye.diff'])

