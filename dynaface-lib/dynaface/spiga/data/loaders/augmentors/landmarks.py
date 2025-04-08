import numpy as np
from PIL import Image
from typing import Any, Dict, Tuple, Union, Optional
from numpy.typing import NDArray

# My libs
import dynaface.spiga.data.loaders.augmentors.utils as dlu


class GeometryBaseAug:
    """
    Base class for geometric augmentations providing common methods for affine transformations.
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Method to apply augmentation. Must be implemented in subclasses.

        Args:
            sample (Dict[str, Any]): A dictionary containing image data and annotations.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Returns:
            Dict[str, Any]: The augmented sample.
        """
        raise NotImplementedError("Inheritance __call__ not defined")

    def map_affine_transformation(
        self,
        sample: Dict[str, Any],
        affine_transf: NDArray[Any],
        new_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Applies an affine transformation to the image, bounding box, and landmarks of the sample.

        Args:
            sample (Dict[str, Any]): A dictionary with keys "image", "bbox", and optionally "landmarks".
            affine_transf (NDArray[Any]): The affine transformation matrix.
            new_size (Optional[Tuple[int, int]], optional): New size for the transformed image. Defaults to None.

        Returns:
            Dict[str, Any]: The sample after applying the affine transformation.
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
        self,
        shape: Tuple[int, int, int, int],
        landmarks: NDArray[Any],
        mask: NDArray[Any],
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        """
        Removes landmarks that fall outside the image bounding box and updates the mask accordingly.

        Args:
            shape (Tuple[int, int, int, int]): The bounding box of the image (x, y, w, h).
            landmarks (NDArray[Any]): Array of landmarks.
            mask (NDArray[Any]): Visibility mask for the landmarks.

        Returns:
            Tuple[NDArray[Any], NDArray[Any]]: Updated mask and landmarks.
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
        affine_transf: NDArray[Any],
        new_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Applies an affine transformation to a PIL image.

        Args:
            image (Image.Image): The original image.
            affine_transf (NDArray[Any]): The affine transformation matrix.
            new_size (Optional[Tuple[int, int]], optional): New image size after transformation. Defaults to None.

        Returns:
            Image.Image: The transformed image.
        """
        if not new_size:
            new_size = image.size
        inv_affine_transf = dlu.get_inverse_transf(affine_transf)
        new_image = image.transform(new_size, Image.AFFINE, inv_affine_transf.flatten())
        return new_image

    def _bbox_affine_trans(
        self, bbox: NDArray[Any], affine_transf: NDArray[Any]
    ) -> NDArray[Any]:
        """
        Applies an affine transformation to a bounding box.

        Args:
            bbox (NDArray[Any]): The bounding box in the format (x, y, w, h).
            affine_transf (NDArray[Any]): The affine transformation matrix.

        Returns:
            NDArray[Any]: The transformed bounding box.
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
        self, landmarks: NDArray[Any], affine_transf: NDArray[Any]
    ) -> NDArray[Any]:
        """
        Applies an affine transformation to landmarks.

        Args:
            landmarks (NDArray[Any]): Array of landmarks.
            affine_transf (NDArray[Any]): The affine transformation matrix.

        Returns:
            NDArray[Any]: The transformed landmarks.
        """
        homog_landmarks = dlu.affine2homogeneous(landmarks)
        new_landmarks = affine_transf.dot(homog_landmarks.T).T
        return new_landmarks


class RSTAug(GeometryBaseAug):
    """
    Augmentation that applies random rotation, scaling, and translation to the image and annotations.
    """

    def __init__(
        self,
        angle_range: float = 45.0,
        scale_min: float = -0.15,
        scale_max: float = 0.15,
        trl_ratio: float = 0.05,
    ) -> None:
        """
        Initialize the RST augmentation.

        Args:
            angle_range (float, optional): Maximum rotation angle in degrees. Defaults to 45.0.
            scale_min (float, optional): Minimum scaling factor. Defaults to -0.15.
            scale_max (float, optional): Maximum scaling factor. Defaults to 0.15.
            trl_ratio (float, optional): Translation ratio relative to the bounding box size. Defaults to 0.05.
        """
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.angle_range = angle_range
        self.trl_ratio = trl_ratio

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply random scale, rotation, and translation to the sample.

        Args:
            sample (Dict[str, Any]): A dictionary containing at least a "bbox" key and image data.

        Returns:
            Dict[str, Any]: The augmented sample.
        """
        x, y, w, h = sample["bbox"]

        # Compute center of the face (bounding box center)
        x0, y0 = (x + w / 2, y + h / 2)

        # Bbox translation
        rnd_Tx = np.random.uniform(-self.trl_ratio, self.trl_ratio) * w
        rnd_Ty = np.random.uniform(-self.trl_ratio, self.trl_ratio) * h
        sample["bbox"][0] += rnd_Tx
        sample["bbox"][1] += rnd_Ty

        scale = 1 + np.random.uniform(self.scale_min, self.scale_max)
        angle = np.random.uniform(-self.angle_range, self.angle_range)

        similarity = dlu.get_similarity_matrix(angle, scale, center=(x0, y0))
        new_sample = self.map_affine_transformation(sample, similarity)
        return new_sample


class TargetCropAug(GeometryBaseAug):
    """
    Augmentation that crops the target region from an image based on a bounding box,
    enlarging it by a specified factor and adjusting image and feature map sizes accordingly.
    """

    def __init__(
        self,
        img_new_size: Union[int, Tuple[int, int]] = 128,
        map_new_size: Union[int, Tuple[int, int]] = 128,
        target_dist: float = 1.3,
    ) -> None:
        """
        Initialize the target crop augmentation.

        Args:
            img_new_size (Union[int, Tuple[int, int]], optional): New size for the cropped image. Defaults to 128.
            map_new_size (Union[int, Tuple[int, int]], optional): New size for the cropped feature map. Defaults to 128.
            target_dist (float, optional): Enlargement factor for the bounding box. Defaults to 1.3.
        """
        self.target_dist = target_dist
        self.new_size_x, self.new_size_y = self._convert_shapes(img_new_size)
        self.map_size_x, self.map_size_y = self._convert_shapes(map_new_size)
        self.img2map_scale = False

        # Mismatch between image shape and feature map shape
        if self.map_size_x != self.new_size_x or self.map_size_y != self.new_size_y:
            self.img2map_scale = True
            self.map_scale_x = self.map_size_x / self.new_size_x
            self.map_scale_y = self.map_size_y / self.new_size_y
            self.map_scale_xx = self.map_scale_x * self.map_scale_x
            self.map_scale_xy = self.map_scale_x * self.map_scale_y
            self.map_scale_yy = self.map_scale_y * self.map_scale_y

    def _convert_shapes(self, new_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """
        Convert a size specification to a tuple (width, height).

        Args:
            new_size (Union[int, Tuple[int, int]]): Either an integer or a tuple of two integers.

        Returns:
            Tuple[int, int]: The width and height.
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
        Crop the target region from the sample based on the bounding box and apply affine transformation.

        Args:
            sample (Dict[str, Any]): A dictionary containing at least "bbox", "image", "landmarks", and "mask_ldm".

        Returns:
            Dict[str, Any]: The sample after cropping and transformation.
        """
        x, y, w, h = sample["bbox"]
        # Enlarge the area around the bounding box
        side = max(w, h) * self.target_dist
        x -= (side - w) / 2
        y -= (side - h) / 2

        # Center of the enlarged bounding box
        x0, y0 = x + side / 2, y + side / 2
        # Scaling factors to match the new image size
        mu_x = self.new_size_x / side
        mu_y = self.new_size_y / side

        new_w = self.new_size_x
        new_h = self.new_size_y
        new_x0, new_y0 = new_w / 2, new_h / 2

        # Create affine transformation for dilation and translation
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
        Rescale landmarks in the sample to adjust for the mismatch between image and feature map sizes.

        Args:
            sample (Dict[str, Any]): The sample containing landmarks and scaling information.

        Returns:
            Dict[str, Any]: The sample with rescaled landmarks.
        """
        lnd_float = sample["landmarks_float"]
        lnd_float[:, 0] = self.map_scale_x * lnd_float[:, 0]
        lnd_float[:, 1] = self.map_scale_y * lnd_float[:, 1]

        # Filter landmarks that fall outside the new feature map dimensions
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
