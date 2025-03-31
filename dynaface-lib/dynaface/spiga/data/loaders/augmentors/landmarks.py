import dynaface.spiga.data.loaders.augmentors.utils as dlu
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional, Tuple, Union


class GeometryBaseAug:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the augmentation to the input sample.

        Args:
            sample (dict): Dictionary containing image, bbox, landmarks, and mask.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Inheritance __call__ not defined")

    def map_affine_transformation(
        self,
        sample: Dict[str, Any],
        affine_transf: np.ndarray,
        new_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Apply affine transformation to the sample's image, bounding box, and landmarks.

        Args:
            sample (dict): Dictionary containing image, bbox, and landmarks.
            affine_transf (np.ndarray): Affine transformation matrix.
            new_size (tuple, optional): New size for the image after transformation.

        Returns:
            dict: Transformed sample.
        """
        sample["image"] = self._image_affine_trans(
            sample["image"], affine_transf, new_size
        )
        sample["bbox"] = self._bbox_affine_trans(sample["bbox"], affine_transf)
        if "landmarks" in sample.keys():
            sample["landmarks"] = self._landmarks_affine_trans(
                sample["landmarks"], affine_transf
            )
        return sample

    def clean_outbbox_landmarks(
        self, shape: Tuple[int, int, int, int], landmarks: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter landmarks that fall outside the bounding box.

        Args:
            shape (tuple): Shape of the bounding box (x, y, w, h).
            landmarks (np.ndarray): Array of landmarks.
            mask (np.ndarray): Mask of landmarks.

        Returns:
            tuple: Updated mask and filtered landmarks.
        """
        filter_x1 = landmarks[:, 0] >= shape[0]
        filter_x2 = landmarks[:, 0] < (shape[0] + shape[2])
        filter_x = np.logical_and(filter_x1, filter_x2)

        filter_y1 = landmarks[:, 1] >= shape[1]
        filter_y2 = landmarks[:, 1] < (shape[1] + shape[3])
        filter_y = np.logical_and(filter_y1, filter_y2)

        filter_bbox = np.logical_and(filter_x, filter_y)
        new_mask = mask * filter_bbox
        new_landmarks = (landmarks.T * new_mask).T
        new_landmarks = new_landmarks.astype(int).astype(float)
        return new_mask, new_landmarks

    def _image_affine_trans(
        self,
        image: Image.Image,
        affine_transf: np.ndarray,
        new_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Apply affine transformation to an image.

        Args:
            image (Image.Image): PIL image.
            affine_transf (np.ndarray): Affine transformation matrix.
            new_size (tuple, optional): New size for the image.

        Returns:
            Image.Image: Transformed image.
        """
        if not new_size:
            new_size = image.size
        inv_affine_transf = dlu.get_inverse_transf(affine_transf)
        new_image = image.transform(new_size, Image.AFFINE, inv_affine_transf.flatten())
        return new_image

    def _bbox_affine_trans(
        self, bbox: Tuple[float, float, float, float], affine_transf: np.ndarray
    ) -> np.ndarray:
        """
        Apply affine transformation to a bounding box.

        Args:
            bbox (tuple): Bounding box (x, y, w, h).
            affine_transf (np.ndarray): Affine transformation matrix.

        Returns:
            np.ndarray: Transformed bounding box.
        """
        x, y, w, h = bbox
        images_bb = []
        for point in ([x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]):
            images_bb.append(affine_transf.dot(point))
        images_bb = np.array(images_bb)

        new_corner0 = np.min(images_bb, axis=0)
        new_corner1 = np.max(images_bb, axis=0)

        new_x, new_y = new_corner0
        new_w, new_h = new_corner1 - new_corner0
        new_bbox = np.array((new_x, new_y, new_w, new_h))
        return new_bbox

    def _landmarks_affine_trans(
        self, landmarks: np.ndarray, affine_transf: np.ndarray
    ) -> np.ndarray:
        """
        Apply affine transformation to landmarks.

        Args:
            landmarks (np.ndarray): Array of landmarks.
            affine_transf (np.ndarray): Affine transformation matrix.

        Returns:
            np.ndarray: Transformed landmarks.
        """
        homog_landmarks = dlu.affine2homogeneous(landmarks)
        new_landmarks = affine_transf.dot(homog_landmarks.T).T
        return new_landmarks


class TargetCropAug(GeometryBaseAug):
    def __init__(
        self,
        img_new_size: Union[int, Tuple[int, int]] = 128,
        map_new_size: Union[int, Tuple[int, int]] = 128,
        target_dist: float = 1.3,
    ):
        """
        Initialize the TargetCropAug with target image size and distance.

        Args:
            img_new_size (int or tuple): Size of the new cropped image.
            map_new_size (int or tuple): Size of the new map.
            target_dist (float): Scaling factor for target distance.
        """
        self.target_dist = target_dist
        self.new_size_x, self.new_size_y = self._convert_shapes(img_new_size)
        self.map_size_x, self.map_size_y = self._convert_shapes(map_new_size)
        self.img2map_scale = False

        # Handle mismatch between image and feature map sizes
        if self.map_size_x != self.new_size_x or self.map_size_y != self.new_size_y:
            self.img2map_scale = True
            self.map_scale_x = self.map_size_x / self.new_size_x
            self.map_scale_y = self.map_size_y / self.new_size_y
            self.map_scale_xx = self.map_scale_x * self.map_scale_x
            self.map_scale_xy = self.map_scale_x * self.map_scale_y
            self.map_scale_yy = self.map_scale_y * self.map_scale_y

    def _convert_shapes(self, new_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """
        Convert shape input to tuple of (width, height).

        Args:
            new_size (int or tuple): Desired size.

        Returns:
            tuple: Converted width and height.
        """
        if isinstance(new_size, (tuple, list)):
            new_size_x = new_size[0]
            new_size_y = new_size[1]
        else:
            new_size_x = new_size
            new_size_y = new_size
        return new_size_x, new_size_y

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply target cropping and affine transformation to the sample.

        Args:
            sample (dict): Dictionary containing image, bbox, landmarks, and mask.

        Returns:
            dict: Transformed sample.
        """
        x, y, w, h = sample["bbox"]
        side = max(w, h) * self.target_dist
        x -= (side - w) / 2
        y -= (side - h) / 2

        # Center of the enlarged bounding box
        x0, y0 = x + side / 2, y + side / 2
        mu_x = self.new_size_x / side
        mu_y = self.new_size_y / side

        new_w = self.new_size_x
        new_h = self.new_size_y
        new_x0, new_y0 = new_w / 2, new_h / 2

        affine_transf = np.array(
            [[mu_x, 0, new_x0 - mu_x * x0], [0, mu_y, new_y0 - mu_y * y0]]
        )

        sample = self.map_affine_transformation(sample, affine_transf, (new_w, new_h))
        if "landmarks" in sample.keys():
            img_shape = np.array([0, 0, self.new_size_x, self.new_size_y])
            sample["landmarks_float"] = sample["landmarks"]
            sample["mask_ldm_float"] = sample["mask_ldm"]
            sample["landmarks"] = np.round(sample["landmarks"])
            sample["mask_ldm"], sample["landmarks"] = self.clean_outbbox_landmarks(
                img_shape, sample["landmarks"], sample["mask_ldm"]
            )

            if self.img2map_scale:
                sample = self._rescale_map(sample)
        return sample

    def _rescale_map(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rescale map landmarks if image and map sizes differ.

        Args:
            sample (dict): Dictionary containing landmarks and mask.

        Returns:
            dict: Updated sample with rescaled landmarks.
        """
        lnd_float = sample["landmarks_float"]
        lnd_float[:, 0] = self.map_scale_x * lnd_float[:, 0]
        lnd_float[:, 1] = self.map_scale_y * lnd_float[:, 1]

        lnd = np.round(lnd_float)
        filter_x = lnd[:, 0] >= self.map_size_x
        filter_y = lnd[:, 1] >= self.map_size_y
        lnd[filter_x] = self.map_size_x - 1
        lnd[filter_y] = self.map_size_y - 1
        new_lnd = (lnd.T * sample["mask_ldm"]).T
        new_lnd = new_lnd.astype(int).astype(float)

        sample["landmarks_float"] = lnd_float
        sample["landmarks"] = new_lnd
        sample["img2map_scale"] = [self.map_scale_x, self.map_scale_y]
        return sample
