import copy
import logging
import os
from typing import Any, Dict, List, Tuple, Union

import dynaface.spiga.inference.pretreatment as pretreat
import numpy as np
import pkg_resources
import torch
from dynaface.spiga.inference.config import ModelConfig
from dynaface.spiga.models.spiga import SPIGA
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Paths
weights_path_dft: str = pkg_resources.resource_filename(
    "dynaface.spiga", "models/weights"
)


class SPIGAFramework:
    def __init__(
        self, model_cfg: ModelConfig, device: torch.device, load3DM: bool = True
    ) -> None:
        """
        Initialize the SPIGAFramework.

        Parameters:
            model_cfg (ModelConfig): Model configuration object.
            device (torch.device): The device on which to run the model.
            load3DM (bool, optional): Flag indicating whether to load the 3D model and camera intrinsic matrix. Defaults to True.
        """
        # Parameters
        self.model_cfg: ModelConfig = model_cfg
        self.device: torch.device = device

        # Pretreatment initialization
        self.transforms = pretreat.get_transformers(self.model_cfg)
        self.transform_batch = pretreat.get_transformers_batch()

        # SPIGA model
        self.model_inputs: List[str] = ["image", "model3d", "cam_matrix"]
        self.model = SPIGA(
            num_landmarks=model_cfg.dataset.num_landmarks,
            num_edges=model_cfg.dataset.num_edges,
        )

        # Load weights and set model
        weights_path: str = self.model_cfg.model_weights_path
        if weights_path is None:
            weights_path = weights_path_dft

        if self.model_cfg.load_model_url:
            model_state_dict = torch.hub.load_state_dict_from_url(
                self.model_cfg.model_weights_url,
                model_dir=weights_path,
                file_name=self.model_cfg.model_weights,
            )
        else:
            weights_file: str = os.path.join(weights_path, self.model_cfg.model_weights)
            model_state_dict = torch.load(weights_file)

        self.model.load_state_dict(model_state_dict)

        # JTH: device support
        self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("SPIGA model loaded")

        # Load 3D model and camera intrinsic matrix
        if load3DM:
            loader_3DM = pretreat.AddModel3D(
                model_cfg.dataset.ldm_ids,
                ftmap_size=model_cfg.ftmap_size,
                focal_ratio=model_cfg.focal_ratio,
                totensor=True,
            )
            params_3DM = self._data2device(loader_3DM())
            self.model3d = params_3DM["model3d"]
            self.cam_matrix = params_3DM["cam_matrix"]

    def inference_batch(
        self, images: List[NDArray[Any]], bbox: List[Any]
    ) -> Dict[str, Any]:
        """
        Perform batch inference on a list of images and bounding boxes.

        Parameters:
            images (List[NDArray[Any]]): List of input images.
            bbox (List[Any]): List of bounding boxes corresponding to each image.

        Returns:
            Dict[str, Any]: Dictionary containing the output features.
        """
        images2: List[Any] = []
        crop_bboxes: List[Any] = []
        for x, y in zip(images, bbox):
            sample: Dict[str, Any] = {"image": x, "bbox": y}
            sample = self.transforms(sample)
            images2.append(sample["image"])
            crop_bboxes.append(sample["bbox"])

        # Images to tensor and device
        batch_images = torch.tensor(np.array(images2), dtype=torch.float)
        batch_images = self._data2device(batch_images)
        # Batch 3D model and camera intrinsic matrix
        batch_model3D = self.model3d.unsqueeze(0).repeat(len(images), 1, 1)
        batch_cam_matrix = self.cam_matrix.unsqueeze(0).repeat(len(images), 1, 1)

        # SPIGA inputs
        model_inputs = [batch_images, batch_model3D, batch_cam_matrix]
        outputs = self.net_forward(model_inputs)
        features = self.postreatment(outputs, crop_bboxes, bbox)
        return features

    def inference(self, image: NDArray[Any], bboxes: List[Any]) -> Dict[str, Any]:
        """
        Perform inference on a single image and its bounding boxes.

        Parameters:
            image (NDArray[Any]): The raw input image.
            bboxes (List[Any]): List of bounding boxes on the image, each defined as [x, y, w, h].

        Returns:
            Dict[str, Any]: Dictionary containing features such as landmarks and headpose.
        """
        batch_crops, crop_bboxes = self.pretreat(image, bboxes)
        outputs = self.net_forward(batch_crops)
        features = self.postreatment(outputs, crop_bboxes, bboxes)
        return features

    def pretreat(
        self, image: NDArray[Any], bboxes: List[Any]
    ) -> Tuple[List[torch.Tensor], List[Any]]:
        """
        Preprocess the image and bounding boxes for inference.

        Parameters:
            image (NDArray[Any]): The raw input image.
            bboxes (List[Any]): List of bounding boxes.

        Returns:
            Tuple[List[torch.Tensor], List[Any]]: A tuple containing a list of preprocessed model inputs and a list of transformed bounding boxes.
        """
        crop_bboxes: List[Any] = []
        crop_images: List[Any] = []
        for bbox in bboxes:
            sample: Dict[str, Any] = {
                "image": copy.deepcopy(image),
                "bbox": copy.deepcopy(bbox),
            }
            sample_crop = self.transforms(sample)
            crop_bboxes.append(sample_crop["bbox"])
            crop_images.append(sample_crop["image"])

        # Images to tensor and device
        batch_images = torch.tensor(np.array(crop_images), dtype=torch.float)
        batch_images = self._data2device(batch_images)
        # Batch 3D model and camera intrinsic matrix
        batch_model3D = self.model3d.unsqueeze(0).repeat(len(bboxes), 1, 1)
        batch_cam_matrix = self.cam_matrix.unsqueeze(0).repeat(len(bboxes), 1, 1)

        # SPIGA inputs
        model_inputs = [batch_images, batch_model3D, batch_cam_matrix]
        return model_inputs, crop_bboxes

    def net_forward(self, inputs: List[torch.Tensor]) -> Any:
        """
        Perform a forward pass through the SPIGA model.

        Parameters:
            inputs (List[torch.Tensor]): Model inputs.

        Returns:
            Any: The raw outputs from the model.
        """
        outputs = self.model(inputs)
        return outputs

    def postreatment(
        self, output: Dict[str, Any], crop_bboxes: List[Any], bboxes: List[Any]
    ) -> Dict[str, Any]:
        """
        Post-process the raw outputs from the model to extract useful features.

        Parameters:
            output (Dict[str, Any]): Raw model outputs.
            crop_bboxes (List[Any]): Transformed bounding boxes from pretreatment.
            bboxes (List[Any]): Original bounding boxes.

        Returns:
            Dict[str, Any]: A dictionary containing processed features (e.g., landmarks, headpose).
        """
        features: Dict[str, Any] = {}
        crop_bboxes = np.array(crop_bboxes)
        bboxes = np.array(bboxes)

        if "Landmarks" in output.keys():
            landmarks = output["Landmarks"][-1].cpu().detach().numpy()
            landmarks = landmarks.transpose((1, 0, 2))
            landmarks = landmarks * self.model_cfg.image_size
            landmarks_norm = (landmarks - crop_bboxes[:, 0:2]) / crop_bboxes[:, 2:4]
            landmarks_out = (landmarks_norm * bboxes[:, 2:4]) + bboxes[:, 0:2]
            landmarks_out = landmarks_out.transpose((1, 0, 2))
            features["landmarks"] = landmarks_out.tolist()

        # Pose output
        if "Pose" in output.keys():
            pose = output["Pose"].cpu().detach().numpy()
            features["headpose"] = pose.tolist()

        return features

    def select_inputs(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Select and process the model inputs from a batch.

        Parameters:
            batch (Dict[str, torch.Tensor]): Dictionary containing input tensors.

        Returns:
            List[torch.Tensor]: List of processed inputs moved to the correct device.
        """
        inputs: List[torch.Tensor] = []
        for ft_name in self.model_inputs:
            data = batch[ft_name]
            inputs.append(self._data2device(data.type(torch.float)))
        return inputs

    def _data2device(
        self, data: Union[torch.Tensor, List[Any], Dict[str, Any]]
    ) -> Union[torch.Tensor, List[Any], Dict[str, Any]]:
        """
        Move data to the designated device.

        Parameters:
            data (Union[torch.Tensor, List[Any], Dict[str, Any]]): The data to be moved.

        Returns:
            Union[torch.Tensor, List[Any], Dict[str, Any]]: Data moved to the device.
        """
        if isinstance(data, list):
            data_var = data
            for data_id, v_data in enumerate(data):
                data_var[data_id] = self._data2device(v_data)
            return data_var
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = self._data2device(v)
            return data
        else:
            with torch.no_grad():
                data_var = data.to(device=self.device, non_blocking=True)
            return data_var
