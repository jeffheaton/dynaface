import csv
import logging
import os
import re
import shutil
import subprocess
import tempfile

import cv2
from dynaface.facial import AnalyzeFace

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


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
            logger.error("Error: Couldn't open the video file.")
            return False

        # Get the frame rate of the video
        self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        self.frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        data_length = round(self.frames / self.frame_rate, 2)
        logger.info(f"Frame rate: {self.frame_rate} FPS")
        logger.info(f"Frames: {self.frames}")
        logger.debug(f"Video length (sec): {data_length}")

        frame_num = 0

        while True:
            ret, frame = cap.read()

            # Break the loop if video has ended
            if not ret:
                break

            frame_num += 1
            output_filename = os.path.join(self.temp_path, f"input-{frame_num}.jpg")
            logger.debug(output_filename)
            # Save the frame as an image
            cv2.imwrite(output_filename, frame)

        # Release the video file
        cap.release()
        return True

    def build(self, output_path, frame_rate):
        if os.path.exists(output_path):
            os.remove(output_path)

        # Filter out the files that start with 'output' and end with '.jpg'
        files = [
            f
            for f in os.listdir(self.temp_path)
            if f.startswith("output") and f.endswith(".jpg")
        ]

        # Sort the files based on the numeric part in their names
        files = sorted(files, key=lambda x: int(x.split("-")[1].split(".jpg")[0]))

        # Read the first image to get the dimensions
        img = cv2.imread(os.path.join(self.temp_path, files[0]))
        height, width, layers = img.shape

        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        # Add images to the video
        for file in files:
            path = os.path.join(self.temp_path, file)
            logger.debug(f"Output: {path}")
            img = cv2.imread(path)

            # Resize image to match the frame size
            resized_img = cv2.resize(img, (width, height))

            out.write(resized_img)

        out.release()
        return True


class VideoToVideo:
    def __init__(self, points, crop):
        self.stats = []
        self.left_area = []
        self.right_area = []
        self.rate = 1 / 240  # 240
        self.data = {}
        self.auto_sync = True
        self._points = points
        self._crop = crop

    def process(self, input_video, output_video, stats=[]):
        # p = ProcessVideoFFMPEG()
        p = ProcessVideoOpenCV()
        if not p.extract(input_video):
            logger.error("Failed to execute ffmpeg")
            return False

        # sample 1st
        filename = os.path.join(p.temp_path, "input-1.jpg")
        logger.debug(filename)
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = AnalyzeFace(stats)
        face.load_image(image, self._crop)

        self.stats = face.get_all_items()
        self.data = {stat: [] for stat in self.stats}

        out_idx = 1
        pupils = None

        idx = 0
        while True:
            idx += 1
            filename = os.path.join(p.temp_path, f"input-{idx}.jpg")
            if not os.path.exists(filename):
                break
            logger.debug(filename)
            # Load frame and crop/size
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face = AnalyzeFace(stats)
            face.load_image(image, self._crop, pupils)

            rec = face.analyze()
            for stat in rec.keys():
                self.data[stat].append(rec[stat])

            face.write_text((10, 30), f"Frame {idx+1}, {round(idx*self.rate*1000)} ms")

            if self._points:
                face.draw_landmarks(numbers=True)

            face.save(os.path.join(p.temp_path, f"output-{out_idx}.jpg"))
            if pupils is None:
                pupils = face.get_pupils()
            out_idx += 1

        p.build(output_video, p.frame_rate)
        p.cleanup()
        logger.debug(p.temp_path)
        return True

    def plot_chart(self, filename, plot_stats=None):
        pass

    def dump_data(self, filename):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            cols = list(self.data.keys())
            writer.writerow(["frame", "time"] + cols)
            data_length = len(self.data[self.stats[0]])
            lst_time = [x * self.rate for x in range(data_length)]

            for i in range(data_length):
                row = [str(i), lst_time[i]]
                for col in cols:
                    row.append(self.data[col][i])
                writer.writerow(row)
