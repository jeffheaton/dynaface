import logging
import os
import shutil
import tempfile

import cv2
import matplotlib.pyplot as plt
from dynaface.facial import AnalyzeFace
from dynaface.video import FrameSeries

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
        self.series = None
        self._points = points
        self._crop = crop

    def process(self, input_video, output_video, stats=[]):
        # p = ProcessVideoFFMPEG()
        p = ProcessVideoOpenCV()
        if not p.extract(input_video):
            logger.error(f"Couldn't read the video {input_video}")
            return False

        # One row per frame, one column per output of the chosen measures.
        self.series = FrameSeries(stats, fps=p.frame_rate)

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

            # Drawn, for the annotated video. A frame without a face adds a row of
            # NaN, which keeps the frames aligned.
            self.series.add(idx - 1, face.analyze())

            face.write_text(
                (10, 30), f"Frame {idx}, {round(self.series.time(idx - 1) * 1000)} ms"
            )

            if self._points:
                face.draw_landmarks(numbers=True)

            face.save(os.path.join(p.temp_path, f"output-{out_idx}.jpg"))
            if pupils is None and not face.is_no_face():
                pupils = face.get_pupils()
            out_idx += 1

        p.build(output_video, p.frame_rate)
        p.cleanup()
        logger.debug(p.temp_path)
        return True

    def plot_chart(self, filename, plot_stats=None):
        """Plots each measure over time, one panel per measure."""
        names = plot_stats or self.series.names
        times = [self.series.time(frame) for frame in self.series.frames]
        fig, axes = plt.subplots(
            len(names), 1, sharex=True, squeeze=False, figsize=(8, 1.5 * len(names))
        )
        for ax, name in zip(axes[:, 0], names):
            ax.plot(times, self.series.values[name])
            ax.set_ylabel(name)
        axes[-1, 0].set_xlabel("Time (seconds)")
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

    def dump_data(self, filename):
        self.series.write_csv(filename)
