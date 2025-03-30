import cv2
import numpy as np
from dynaface.spiga.data.loaders.transforms import AddModel3D, TargetCrop, ToOpencv
from PIL import Image
from torchvision import transforms


def get_transformers(data_config):
    transformer_seq = [
        Opencv2Pil(),
        TargetCrop(data_config.image_size, data_config.target_dist),
        ToOpencv(),
        NormalizeAndPermute(),
    ]
    return transforms.Compose(transformer_seq)


def get_transformers_batch():
    transformer_seq = [
        NormalizeAndPermute(),
    ]
    return transforms.Compose(transformer_seq)


class NormalizeAndPermute:
    def __call__(self, sample):
        image = np.array(sample["image"], dtype=float)
        image = np.transpose(image, (2, 0, 1))
        sample["image"] = image / 255
        return sample


class Opencv2Pil:
    def __call__(self, sample):
        image = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB)
        sample["image"] = Image.fromarray(image)
        return sample
