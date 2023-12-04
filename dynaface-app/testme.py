import logging
import time

from facial_analysis.facial import load_face_image
from PyQt6.QtCore import QThread, pyqtSignal
from facenet_pytorch import MTCNN
from facial_analysis.image import ImageAnalysis, load_image
from facial_analysis.spiga.inference.config import ModelConfig
from facial_analysis.spiga.inference.framework import SPIGAFramework

SOURCE_IMG = "/Users/jeff/data/facial/samples/2021-8-19.png"
DEST_IMG = "/Users/jeff/data/facial/samples/2021-8-19-output.png"

SPIGA_MODEL = "wflw"

logger = logging.getLogger(__name__)
t = None


class Worker(QThread):
    # Signal to update the GUI
    # update_signal = pyqtSignal(str)

    def run(self):
        # run_simple_test()
        # run_ai_test()
        run_direct_test()

        # self.update_signal.emit("Task completed!")


def run_simple_test():
    for i in range(50):
        time.sleep(1)  # Simulate a task taking some time
        logger.info(f"Running... {i+1}")


def run_ai_test():
    face = load_face_image(SOURCE_IMG, device="cpu")
    face.analyze()
    face.draw_landmarks(numbers=True)
    face.save(DEST_IMG)
    logger.info("Test done")


def run_direct_test():
    logger.info("Beginning direct test")
    device = "cpu"
    img = load_image(SOURCE_IMG)
    mtcnn = MTCNN(keep_all=True, device=device)
    bbox, _ = mtcnn.detect(img)
    logger.info(bbox)
    config = ModelConfig(dataset_name=SPIGA_MODEL, load_model_url=False)
    processor = SPIGAFramework(config, device=device)
    bbox = bbox[0]
    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    features = processor.inference(img, [bbox])
    logger.info(features)

    logger.info("Ending direct test")


def test_it():
    global t
    logger.info("Running test")

    t = Worker()
    # self.thread.update_signal.connect(self.updateLabel)
    t.start()
