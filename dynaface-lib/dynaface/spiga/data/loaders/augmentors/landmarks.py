# My libs
import dynaface.spiga.data.loaders.augmentors.utils as dlu
import numpy as np
from PIL import Image


class GeometryBaseAug:
    def __call__(self, sample):
        raise NotImplementedError("Inheritance __call__ not defined")

    def map_affine_transformation(self, sample, affine_transf, new_size=None):
        sample["image"] = self._image_affine_trans(
            sample["image"], affine_transf, new_size
        )
        sample["bbox"] = self._bbox_affine_trans(sample["bbox"], affine_transf)
        if "landmarks" in sample.keys():
            sample["landmarks"] = self._landmarks_affine_trans(
                sample["landmarks"], affine_transf
            )
        return sample

    def clean_outbbox_landmarks(self, shape, landmarks, mask):
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

    def _image_affine_trans(self, image, affine_transf, new_size=None):
        if not new_size:
            new_size = image.size
        inv_affine_transf = dlu.get_inverse_transf(affine_transf)
        new_image = image.transform(new_size, Image.AFFINE, inv_affine_transf.flatten())
        return new_image

    def _bbox_affine_trans(self, bbox, affine_transf):
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

    def _landmarks_affine_trans(self, landmarks, affine_transf):
        homog_landmarks = dlu.affine2homogeneous(landmarks)
        new_landmarks = affine_transf.dot(homog_landmarks.T).T
        return new_landmarks


class TargetCropAug(GeometryBaseAug):
    def __init__(self, img_new_size=128, map_new_size=128, target_dist=1.3):
        self.target_dist = target_dist
        self.new_size_x, self.new_size_y = self._convert_shapes(img_new_size)
        self.map_size_x, self.map_size_y = self._convert_shapes(map_new_size)
        self.img2map_scale = False

        # Mismatch between img shape and featuremap shape
        if self.map_size_x != self.new_size_x or self.map_size_y != self.new_size_y:
            self.img2map_scale = True
            self.map_scale_x = self.map_size_x / self.new_size_x
            self.map_scale_y = self.map_size_y / self.new_size_y
            self.map_scale_xx = self.map_scale_x * self.map_scale_x
            self.map_scale_xy = self.map_scale_x * self.map_scale_y
            self.map_scale_yy = self.map_scale_y * self.map_scale_y

    def _convert_shapes(self, new_size):
        if isinstance(new_size, (tuple, list)):
            new_size_x = new_size[0]
            new_size_y = new_size[1]
        else:
            new_size_x = new_size
            new_size_y = new_size
        return new_size_x, new_size_y

    def __call__(self, sample):
        x, y, w, h = sample["bbox"]
        # we enlarge the area taken around the bounding box
        # it is neccesary to change the botton left point of the bounding box
        # according to the previous enlargement. Note this will NOT be the new
        # bounding box!
        # We return square images, which is neccesary since
        # all the images must have the same size in order to form batches
        side = max(w, h) * self.target_dist
        x -= (side - w) / 2
        y -= (side - h) / 2

        # center of the enlarged bounding box
        x0, y0 = x + side / 2, y + side / 2
        # homothety factor, chosen so the new horizontal dimension will
        # coincide with new_size
        mu_x = self.new_size_x / side
        mu_y = self.new_size_y / side

        # new_w, new_h = new_size, int(h * mu)
        new_w = self.new_size_x
        new_h = self.new_size_y
        new_x0, new_y0 = new_w / 2, new_h / 2

        # dilatation + translation
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

    def _rescale_map(self, sample):
        # Rescale
        lnd_float = sample["landmarks_float"]
        lnd_float[:, 0] = self.map_scale_x * lnd_float[:, 0]
        lnd_float[:, 1] = self.map_scale_y * lnd_float[:, 1]

        # Filter landmarks
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
