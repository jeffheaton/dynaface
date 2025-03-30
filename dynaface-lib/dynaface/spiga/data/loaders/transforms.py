import cv2
import numpy as np
import torch
from dynaface.spiga.data.loaders.augmentors.landmarks import (
    TargetCropAug,
)
from dynaface.spiga.data.loaders.augmentors.modern_posit import PositPose


class ToOpencv:
    def __call__(self, sample):
        # Convert in a numpy array and change to GBR
        image = np.array(sample["image"])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sample["image"] = image
        return sample


class TargetCrop(TargetCropAug):
    def __init__(self, crop_size=256, target_dist=1.6):
        super(TargetCrop, self).__init__(crop_size, crop_size, target_dist)


class AddModel3D(PositPose):
    def __init__(self, ldm_ids, ftmap_size=(256, 256), focal_ratio=1.5, totensor=False):
        super(AddModel3D, self).__init__(ldm_ids, focal_ratio=focal_ratio)
        img_bbox = [
            0,
            0,
            ftmap_size[1],
            ftmap_size[0],
        ]  # Shapes given are inverted (y,x)
        self.cam_matrix = self._camera_matrix(img_bbox)

        if totensor:
            self.cam_matrix = torch.tensor(self.cam_matrix, dtype=torch.float)
            self.model3d_world = torch.tensor(self.model3d_world, dtype=torch.float)

    def __call__(self, sample={}):
        # Save intrinsic matrix and 3D model landmarks
        sample["cam_matrix"] = self.cam_matrix
        sample["model3d"] = self.model3d_world
        return sample
