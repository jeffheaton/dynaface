import os

import torch
from facenet_pytorch import MTCNN
from facenet_pytorch.models.mtcnn import ONet, PNet, RNet
from torch import nn


class MTCNN2(MTCNN):
    def __init__(
        self,
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        select_largest=True,
        selection_method=None,
        keep_all=False,
        device=None,
        path=None,
    ):
        nn.Module.__init__(self)

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.selection_method = selection_method

        self.pnet = PNet(pretrained=False)
        self.load_weights(self.pnet, os.path.join(path, "pnet.pt"))
        self.rnet = RNet(pretrained=False)
        self.load_weights(self.rnet, os.path.join(path, "rnet.pt"))
        self.onet = ONet(pretrained=False)
        self.load_weights(self.onet, os.path.join(path, "onet.pt"))

        self.device = torch.device("cpu")
        if device is not None:
            self.device = device
            self.to(device)

        if not self.selection_method:
            self.selection_method = "largest" if self.select_largest else "probability"

        # pnet.pt

    def load_weights(self, net, filename):
        state_dict = torch.load(filename)
        net.load_state_dict(state_dict)


class FindFace:
    mtcnn = None

    def init(device=None, path=None):
        if device is None:
            has_mps = torch.backends.mps.is_built()
            device = "mps" if has_mps else "gpu" if torch.cuda.is_available() else "cpu"

        if device == "mps":
            device = "cpu"  # seems to be some mps issue we need to look at

        if path is None:
            FindFace.mtcnn = MTCNN(keep_all=True, device=device)
        else:
            FindFace.mtcnn = MTCNN2(keep_all=True, device=device, path=path)

    def is_init():
        return not FindFace.mtcnn is None

    def detect_face(img):
        if FindFace.mtcnn is None:
            FindFace.init()
        boxes, _ = FindFace.mtcnn.detect(img)
        if boxes is None:
            return None
        return boxes[0]

    def crop(img):
        if FindFace.mtcnn is None:
            FindFace.init()
        boxes, _ = FindFace.mtcnn.detect(img)
        if boxes is not None:
            # Assuming the first face detected
            box = boxes[0]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            return img[y1:y2, x1:x2]

        return None
