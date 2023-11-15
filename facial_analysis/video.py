import subprocess
import tempfile
import shutil
import os
import re
import plotly.graph_objects as go
import plotly.io as pio
from facial_analysis.facial import AnalyzeFace, CropImage
import cv2
import csv

SAMPLE_RATE = 44100

class ProcessVideoFFMPEG:
  def __init__(self):
    # Create a temporary directory
    self.temp_path = tempfile.mkdtemp()
    self.temp_output = os.path.join(self.temp_path, "output.txt")
    self.input_images = os.path.join(self.temp_path,'input-%d.jpg')
    self.output_images = os.path.join(self.temp_path,'output-%d.jpg') 

  def execute_command(self, cmd):
    with open(self.temp_output, 'w') as fp:
      result = subprocess.call(cmd, shell=True, stdout=fp)

    if result!=0:
      return None

    with open(self.temp_output, 'r') as fp:
      result = fp.read()

    print(f"Executed command: {cmd}, result:")
    print(result)
    print("---------\n")
    return result.split('\n')

  def cleanup(self):
    # Delete the temporary directory when you're done
    shutil.rmtree(self.temp_path)

  def extract(self, input_file, quality=2):
    # Delete audio file if it already exists
    self.audio_file = os.path.join(self.temp_path,'audio.wav')

    if os.path.exists(self.audio_file):
      os.remove(self.audio_file)

    # First call to ffmpeg extracts the video image frames
    results = self.execute_command(f"ffmpeg -i {input_file} -qscale:v {quality} {self.input_images} -hide_banner 2>&1")
    if results is None:
      return False

    # Second call to ffmpeg extracts the audio.  We also attempt to get the FPS from
    # this call.
    print(f"ffmpeg -i {input_file} -ab 160k -ac 2 -ar {SAMPLE_RATE} -vn {input_file} 2>&1")
    results = self.execute_command(f"ffmpeg -i {input_file} -ab 160k -ac 2 -ar {SAMPLE_RATE} -vn {self.audio_file} 2>&1")

    self.frame_rate = 30 # default, but try to detect
    for line in results:
      m = re.search('Stream #.*Video.* ([0-9]*) fps',line)
      if m is not None:
        self.frame_rate = float(m.group(1))
        print(f"Detected framerate of {self.frame_rate}")

    # Report on the frame rate and attempt to obtain audio sample rate.
    print(f"Frame rate used: {self.frame_rate}")
    print(self.temp_path)
    return True

  def build(self, output_path, frame_rate):
    if os.path.exists(output_path):
      os.remove(output_path)
    self.execute_command(f"ffmpeg -framerate {frame_rate} -i {self.output_images} -i {self.audio_file} -strict -2 {output_path} 2>&1")

class ProcessVideoOpenCV:
  def __init__(self):
    # Create a temporary directory
    self.temp_path = tempfile.mkdtemp()

  def cleanup(self):
    # Delete the temporary directory when you're done
    shutil.rmtree(self.temp_path)
    
  def extract(self, input_file, quality=2):
    
    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Check if video file opened successfully
    if not cap.isOpened():
      print("Error: Couldn't open the video file.")
      exit()

    # Get the frame rate of the video
    self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Frame rate: {self.frame_rate} FPS")

    frame_num = 0

    while True:
      ret, frame = cap.read()

      # Break the loop if video has ended
      if not ret:
        break

      frame_num += 1
      output_filename = os.path.join(self.temp_path, f"input-{frame_num}.jpg")
      print(output_filename)
      # Save the frame as an image
      cv2.imwrite(output_filename, frame)

    # Release the video file
    cap.release()
    return True
  
  def build(self, output_path, frame_rate):
    if os.path.exists(output_path):
      os.remove(output_path)

    # Filter out the files that start with 'output' and end with '.jpg'
    files = [f for f in os.listdir(self.temp_path) if f.startswith('output') and f.endswith('.jpg')]

    # Sort the files based on the numeric part in their names
    files = sorted(files, key=lambda x: int(x.split('-')[1].split('.jpg')[0]))

    # Read the first image to get the dimensions
    img = cv2.imread(os.path.join(self.temp_path, files[0]))
    height, width, layers = img.shape

    # Create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # Add images to the video
    for file in files:
        path = os.path.join(self.temp_path, file)
        print(f"Output: {path}")
        img = cv2.imread(path)

        # Resize image to match the frame size
        resized_img = cv2.resize(img, (width, height))

        out.write(resized_img)

    out.release()
    return True
  


class VideoToVideo:
  def __init__(self):
    self.stats = []
    self.left_area = []
    self.right_area = []
    self.rate = 1/240 # 240
    self.data = {}
    self.auto_sync = True

  def process(self, input_video, output_video, stats=[]):
    #p = ProcessVideoFFMPEG()
    p = ProcessVideoOpenCV()
    if not p.extract(input_video):
      print("Failed to execute ffmpeg")
      return False

    # sample 1st
    filename = os.path.join(p.temp_path,f"input-1.jpg")
    print(filename)
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = AnalyzeFace(stats)
    face.load_image(image)

    self.stats = face.get_all_stats()    
    self.data = {stat: [] for stat in self.stats}
   
    out_idx = 1

    idx = 0
    while True:
      idx+=1
      filename = os.path.join(p.temp_path,f"input-{idx}.jpg")
      if not os.path.exists(filename): break
      print(filename)
      # Load frame and crop/size
      image = cv2.imread(filename)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = CropImage.crop(image)
      face = AnalyzeFace(stats)
      face.load_image(image)
      
      rec = face.analyze()
      for stat in rec.keys():
        self.data[stat].append(rec[stat])
      
      face.write_text((10,30), f"Frame {idx+1}, {round(idx*self.rate*1000)} ms")
      face.save(os.path.join(p.temp_path,f"output-{out_idx}.jpg"))
      out_idx += 1

    p.build(output_video,p.frame_rate)
    p.cleanup()
    print(p.temp_path)
    return True

  def plot_chart(self, filename, plot_stats=None):
    if plot_stats is None:
      plot_stats = self.data.keys()
    # create time axis
    l = len(self.data[self.stats[0]])
    lst_time = [x*self.rate for x in range(l)]

    layout = go.Layout(
        title=f"Blink Efficency",
        autosize=False,
        width=1500,
        height=540,
        xaxis_title="Time (s)",
        xaxis_showticklabels = True,
        #yaxis_title="Area (mm^2)",
        yaxis_title="Value (multiple units)",
        yaxis_showticklabels = True)

    fig = go.Figure(layout=layout)

    for stat in self.data.keys():
      if stat in plot_stats:
        fig.add_trace(go.Scatter(x=lst_time, y=self.data[stat], name=stat, )) # line=dict(color='red'))

    #fig.add_trace(go.Scatter(x=lst_time, y=self.right_area, name="Right Area", line=dict(color='blue')))
    fig.write_image(filename)

  def dump_data(self, filename):
    with open(filename, 'w') as f:
      writer = csv.writer(f)
      cols = list(self.data.keys())
      writer.writerow(["frame","time"]+cols)
      l = len(self.data[self.stats[0]])
      lst_time = [x*self.rate for x in range(l)]

      for i in range(l):
        row = [str(i),lst_time[i]]
        for col in cols:
          row.append(self.data[col][i])
        writer.writerow(row)


