"""
DynafaceOnnxInference — unified ONNX Runtime wrapper for the three models
used in the face pipeline:

    1. BlazeFace short-range face detector  -> find a face's bounding box
    2. SPIGA WFLW landmark model            -> 98 facial landmarks + pose
    3. U^2-Net                              -> background removal mask

Usage:
    from dynaface.dynaface_onnx import DynafaceOnnxInference

    dynaface = DynafaceOnnxInference('.', device='cuda')

    box, score = dynaface.detect_face(image_bgr)          # (x, y, w, h), float
    landmarks, pose = dynaface.find_landmarks(image_bgr, box)  # [98, 2], [6]
    rgba = dynaface.remove_background(image_bgr)          # BGRA, alpha = mask
"""

import math
import os
from typing import List, Optional, Tuple, cast

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# WFLW 98-landmark 3D template and camera intrinsics (ftmap 64x64, focal 1.5)
# ---------------------------------------------------------------------------
_MODEL3D = np.array(
    [
        [-0.853184236268, 0.710460614932, -0.393345437621],
        [-0.865903665506, 0.697935135807, -0.281548075586],
        [-0.878623094748, 0.685409656727, -0.169750713576],
        [-0.90048907572, 0.646541048786, -0.054976885980],
        [-0.92235505661, 0.607672440835, 0.059796941671],
        [-0.921595827864, 0.576668701754, 0.161148197750],
        [-0.920836599128, 0.545664962608, 0.262499453743],
        [-0.891987904048, 0.495939744597, 0.350772314723],
        [-0.863139209001, 0.446214526602, 0.439045175772],
        [-0.796026245014, 0.390325238349, 0.502547532515],
        [-0.728913281017, 0.334435950088, 0.566049889293],
        [-0.657186170535, 0.285302127936, 0.609874056230],
        [-0.585459059997, 0.236168305748, 0.653698223146],
        [-0.536840395466, 0.182237836481, 0.682871772143],
        [-0.488221730878, 0.128307367184, 0.712045321068],
        [-0.474793009911, 0.065927929287, 0.725020759991],
        [-0.461364289002, 0.003548491464, 0.737996198981],
        [-0.463813244263, -0.063825309161, 0.735566080866],
        [-0.466262199443, -0.131199109787, 0.733135962680],
        [-0.513521912927, -0.194286440465, 0.699811948342],
        [-0.560781626342, -0.257373771199, 0.666487934013],
        [-0.613723257723, -0.308598755049, 0.618041459290],
        [-0.666664889106, -0.359823738861, 0.569594984479],
        [-0.714432235344, -0.407851339087, 0.522433885778],
        [-0.762199581605, -0.455878939366, 0.475272786996],
        [-0.828822422342, -0.519515088045, 0.365356986341],
        [-0.895445263130, -0.583151236655, 0.255441185628],
        [-0.910235122162, -0.609093253652, 0.166017120718],
        [-0.925024981150, -0.635035270609, 0.076593055883],
        [-0.915635837119, -0.661799095895, -0.025251322986],
        [-0.906246693121, -0.688562921268, -0.127095701883],
        [-0.904118559673, -0.700222989101, -0.229632887961],
        [-0.901990426154, -0.711883056935, -0.332170073985],
        [-0.466456539275, 0.552168877879, -0.483792334207],
        [-0.308160933540, 0.459810924980, -0.561780416182],
        [-0.216753374882, 0.360178576451, -0.566290707559],
        [-0.148694799479, 0.249770054109, -0.530336745190],
        [-0.124216132105, 0.101941089981, -0.482471777751],
        [-0.132953775070, 0.100790757462, -0.429593539303],
        [-0.157432442443, 0.248619721590, -0.477458506742],
        [-0.225491017847, 0.359028243932, -0.513412469111],
        [-0.316898576504, 0.458660592461, -0.508902177734],
        [-0.131673602172, -0.097976488432, -0.465935806417],
        [-0.135680573937, -0.214281256932, -0.497130488240],
        [-0.223000042464, -0.373011222167, -0.520379035114],
        [-0.319251580460, -0.501607216528, -0.500371303275],
        [-0.465084060803, -0.573625223791, -0.457842172777],
        [-0.310513937496, -0.500456884008, -0.447493064827],
        [-0.231737685429, -0.374161554686, -0.467500796666],
        [-0.144418216902, -0.215431589451, -0.444252249792],
        [-0.140411245136, -0.099126820952, -0.413057567969],
        [-0.139142361862, -0.008435261941, -0.416310824526],
        [-0.106471830333, -0.007312178351, -0.267155736618],
        [-0.054453930590, -0.001901740064, -0.142359799356],
        [0.0, 0.0, 0.0],
        [-0.235537796980, 0.118903311783, 0.079278454225],
        [-0.171210015652, 0.050430024294, 0.104665185108],
        [-0.154139340027, -0.003126570926, 0.111845126622],
        [-0.173275169304, -0.068754398342, 0.099545856333],
        [-0.238513462350, -0.148988810614, 0.073784851484],
        [-0.383328256738, 0.406302776084, -0.360389456414],
        [-0.310253684249, 0.344774145080, -0.384294131116],
        [-0.298449428843, 0.303148027928, -0.398970401478],
        [-0.286645173373, 0.261521910707, -0.396020592392],
        [-0.330243237010, 0.160787895283, -0.371046071665],
        [-0.323105980144, 0.245621576158, -0.334421719595],
        [-0.331524487470, 0.286856859665, -0.324384347457],
        [-0.339942994736, 0.328092143155, -0.331973054752],
        [-0.314140711048, -0.163880511996, -0.340747336431],
        [-0.279030578852, -0.250844605212, -0.392490504075],
        [-0.282436250152, -0.295807479997, -0.394548614690],
        [-0.285841921386, -0.340770354713, -0.378980645804],
        [-0.377158168879, -0.432352000574, -0.348665008377],
        [-0.318530865652, -0.333467711265, -0.328033806608],
        [-0.315834032602, -0.290044551442, -0.322730607606],
        [-0.313137199533, -0.246621391603, -0.335053488012],
        [-0.354306324759, 0.222983201100, 0.277627584818],
        [-0.270218764688, 0.157062857603, 0.248737641675],
        [-0.225237980339, 0.081635485618, 0.236184984280],
        [-0.212609820801, 0.003915777595, 0.244998123230],
        [-0.208501270490, -0.062942106644, 0.230774971513],
        [-0.235471195707, -0.135507933036, 0.240646328002],
        [-0.332281891846, -0.240381217797, 0.283709533359],
        [-0.283978333058, -0.143706630542, 0.329461104519],
        [-0.256464077312, -0.084139951946, 0.355354733250],
        [-0.250423014186, -0.008137040189, 0.360357397838],
        [-0.269661366141, 0.077826069065, 0.349041413147],
        [-0.322383820174, 0.157111645515, 0.311971743235],
        [-0.341170846657, 0.182660017758, 0.278377925956],
        [-0.270854173882, 0.075943475389, 0.283519721316],
        [-0.254428375697, 0.002670132255, 0.286917503124],
        [-0.257845545665, -0.068506200339, 0.285630836758],
        [-0.314605542850, -0.194170296105, 0.279641061938],
        [-0.258696545260, -0.068986247483, 0.288076969827],
        [-0.256229467080, 0.002739134144, 0.289157237470],
        [-0.273027820052, 0.076698969737, 0.286311968999],
        [-0.328919887677, 0.291183407723, -0.363024170951],
        [-0.314639907590, -0.294656095888, -0.353995131537],
    ],
    dtype=np.float32,
)  # [98, 3]

_CAM_MATRIX = np.array(
    [
        [96.0, 0.0, 32.0],
        [0.0, 96.0, 32.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)  # [3, 3]

# SPIGA was trained with a crop that pads the detector bbox to a square
# `max(w, h) * _SPIGA_TARGET_DIST` region before resizing to _SPIGA_CROP_SIZE.
# This must match dynaface's own TargetCropAug(crop_size=256, target_dist=1.6).
_SPIGA_TARGET_DIST = 1.6
_SPIGA_CROP_SIZE = 256

# Maps dynaface's device strings ("cpu"/"cuda"/"mps") to ONNX Runtime
# execution providers, ordered with a CPU fallback in case the accelerated
# provider isn't available in the installed onnxruntime package.
_DEVICE_PROVIDERS = {
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "mps": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


def _resolve_providers(device: str) -> List[str]:
    return _DEVICE_PROVIDERS.get(device.lower(), ["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# BlazeFace (short-range) anchor generation — SSD-style anchors matching the
# model's fixed config: 4 layers, strides 8/16/16/16, 128x128 input,
# fixed_anchor_size, giving 896 anchors total.
# ---------------------------------------------------------------------------
def _generate_blaze_face_anchors() -> NDArray[np.float32]:
    # These constants mirror the fixed SSD anchor config baked into the
    # BlazeFace short-range TFLite model (MediaPipe's ssd_anchors_calculator).
    # The ONNX export only gives us raw regressors/scores, not decoded boxes,
    # so the anchor grid has to be reconstructed here to match it exactly.
    strides = [8, 16, 16, 16]
    input_size = 128
    min_scale, max_scale = 0.1484375, 0.75
    num_layers = len(strides)

    def scale_at(i: int) -> float:
        return min_scale + (max_scale - min_scale) * i / (num_layers - 1)

    anchors = []
    layer_id = 0
    while layer_id < num_layers:
        # Consecutive layers that share a stride (here: the three stride-16
        # layers) are treated as one group and get their scales concatenated,
        # so each grid cell in that group ends up with all of their anchors.
        last = layer_id
        scales = []
        while last < num_layers and strides[last] == strides[layer_id]:
            scale = scale_at(last)
            scale_next = 1.0 if last == num_layers - 1 else scale_at(last + 1)
            # Two anchors per layer: one at this layer's scale, one at the
            # geometric mean with the next layer's scale (aspect_ratios=[1.0]
            # plus interpolated_scale_aspect_ratio=1.0 in the original config).
            scales.append(scale)
            scales.append(math.sqrt(scale * scale_next))
            last += 1

        stride = strides[layer_id]
        feat = input_size // stride
        for y in range(feat):
            for x in range(feat):
                for _ in scales:
                    # fixed_anchor_size=true in the source config: every
                    # anchor is given a unit width/height regardless of its
                    # scale, so w/h below are placeholders decode() divides
                    # out symmetrically rather than real box dimensions.
                    anchors.append([(x + 0.5) / feat, (y + 0.5) / feat, 1.0, 1.0])
        layer_id = last

    return np.array(anchors, dtype=np.float32)  # [896, 4] (x_ctr, y_ctr, w, h)


class DynafaceOnnxInference:
    """
    Loads the face detector, landmark, and background-removal ONNX models
    from a single directory and exposes one inference call per model.
    """

    def __init__(
        self,
        model_dir: str = ".",
        face_model: str = "blaze_face_short_range.onnx",
        landmark_model: str = "spiga_wflw.onnx",
        background_model: str = "u2net.onnx",
        device: str = "cpu",
    ) -> None:
        providers = _resolve_providers(device)
        self._face_session = ort.InferenceSession(
            os.path.join(model_dir, face_model), providers=providers
        )
        self._landmark_session = ort.InferenceSession(
            os.path.join(model_dir, landmark_model), providers=providers
        )
        self._background_session = ort.InferenceSession(
            os.path.join(model_dir, background_model), providers=providers
        )

        # Read the actual input/output names from the ONNX graph instead of
        # hardcoding them: exported models often have auto-generated names
        # (e.g. u2net's output is named "1959", not something readable).
        self._face_input_name = self._face_session.get_inputs()[0].name
        self._face_output_names = [o.name for o in self._face_session.get_outputs()]

        self._bg_input_name = self._background_session.get_inputs()[0].name
        self._bg_output_name = self._background_session.get_outputs()[0].name

        self._face_anchors = _generate_blaze_face_anchors()
        # Pre-batch SPIGA's fixed auxiliary inputs (batch size is always 1).
        self._model3d = _MODEL3D[None]
        self._cam_matrix = _CAM_MATRIX[None]

    # -- face detection -------------------------------------------------------

    def detect_face(
        self, image_bgr: NDArray[np.uint8], score_thresh: float = 0.5
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[float]]:
        """
        Find the most confident face in image_bgr.

        Returns ((x, y, w, h), score) in pixel coordinates of image_bgr, or
        (None, None) if no face scores above score_thresh. BlazeFace's score
        is not calibrated the same way as other detectors (e.g. MTCNN); tune
        score_thresh against your own data rather than reusing a threshold
        from a different detector.
        """
        h, w = image_bgr.shape[:2]
        tensor = self._preprocess_face(image_bgr)

        regressors, scores = self._face_session.run(
            self._face_output_names, {self._face_input_name: tensor}
        )
        regressors = regressors[0]  # [896, 16]: 4 box params + 6 keypoints * 2
        scores = scores[0, :, 0]  # [896] raw confidence logits

        # Model outputs raw logits (float32), not probabilities. Compute the
        # sigmoid in float64: exp(100) alone overflows float32's range.
        scores = 1.0 / (
            1.0 + np.exp(-np.clip(scores.astype(np.float64), -100.0, 100.0))
        )
        keep = scores > score_thresh
        if not np.any(keep):
            return None, None

        boxes = self._decode_face_boxes(regressors[keep], self._face_anchors[keep])
        scores = scores[keep]

        # Collapse overlapping detections down to the single best face.
        order = self._nms(boxes, scores)
        if len(order) == 0:
            return None, None
        best = order[0]
        y_min, x_min, y_max, x_max = boxes[best]  # normalized to [0, 1]

        x0 = max(0, int(x_min * w))
        y0 = max(0, int(y_min * h))
        x1 = min(w, int(x_max * w))
        y1 = min(h, int(y_max * h))
        return (x0, y0, x1 - x0, y1 - y0), float(scores[best])

    @staticmethod
    def _preprocess_face(image_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (128, 128))
        # BlazeFace expects pixels normalized to [-1, 1], not [0, 1].
        tensor = (rgb.astype(np.float32) / 127.5) - 1.0
        return tensor[None]  # [1, 128, 128, 3]

    @staticmethod
    def _decode_face_boxes(
        raw_boxes: NDArray[np.float32], anchors: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """raw_boxes: [N, 16] -> boxes: [N, 4] normalized (ymin, xmin, ymax, xmax)."""
        # Standard SSD/BlazeFace box decode: the model predicts offsets in
        # 128px-input units relative to each anchor's center, scaled by the
        # anchor's own width/height (here always 1.0, see fixed_anchor_size
        # note above), not absolute coordinates.
        scale = 128.0
        x_center = raw_boxes[:, 0] / scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[:, 1] / scale * anchors[:, 3] + anchors[:, 1]
        box_w = raw_boxes[:, 2] / scale * anchors[:, 2]
        box_h = raw_boxes[:, 3] / scale * anchors[:, 3]

        return np.stack(
            [
                y_center - box_h / 2.0,
                x_center - box_w / 2.0,
                y_center + box_h / 2.0,
                x_center + box_w / 2.0,
            ],
            axis=1,
        )

    @staticmethod
    def _nms(
        boxes: NDArray[np.float32], scores: NDArray[np.float32], iou_thresh: float = 0.3
    ) -> List[int]:
        """Greedy non-max suppression; returns indices to keep, highest score first."""
        order = scores.argsort()[::-1]
        y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        keep = []
        while order.size > 0:
            # Take the highest-scoring remaining box, then drop every other
            # box that overlaps it too much (likely the same face).
            i = order[0]
            keep.append(int(i))
            rest = order[1:]

            yy1 = np.maximum(y1[i], y1[rest])
            xx1 = np.maximum(x1[i], x1[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            xx2 = np.minimum(x2[i], x2[rest])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

            order = rest[iou <= iou_thresh]
        return keep

    # -- landmarks --------------------------------------------------------------

    def find_landmarks(
        self,
        image_bgr: NDArray[np.uint8],
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Predict 98 WFLW landmarks and head pose for the face at bbox.

        Parameters
        ----------
        image_bgr : np.ndarray
            The full (uncropped) BGR image.
        bbox : (x, y, w, h)
            Face bounding box in image_bgr pixel coordinates, e.g. from
            detect_face(). Internally padded to SPIGA's training-time crop
            (a square region of max(w, h) * _SPIGA_TARGET_DIST, centered on
            the box) before being resized to _SPIGA_CROP_SIZE.

        Returns
        -------
        landmarks : np.ndarray, shape (98, 2), float32, pixel coordinates of
            image_bgr.
        pose : np.ndarray, shape (6,), float32, raw [euler(3), translation(3)]
            head pose output, passed through unchanged (matches dynaface's
            existing SPIGAFramework "headpose" convention).
        """
        tensor, crop_affine = self._preprocess_landmarks(image_bgr, bbox)

        landmarks_norm, pose = self._landmark_session.run(
            ["landmarks", "pose"],
            {
                "image": tensor,
                "model3d": self._model3d,
                "cam_matrix": self._cam_matrix,
            },
        )
        landmarks_crop_px = landmarks_norm[0] * float(_SPIGA_CROP_SIZE)  # [98, 2]

        # Map landmarks from the crop's pixel space back to image_bgr's.
        inv_affine = cv2.invertAffineTransform(crop_affine)
        ones = np.ones((landmarks_crop_px.shape[0], 1), dtype=np.float32)
        landmarks_h = np.hstack([landmarks_crop_px, ones])
        landmarks_img = landmarks_h @ inv_affine.T

        return landmarks_img.astype(np.float32), pose[0].astype(np.float32)

    @staticmethod
    def _crop_affine(
        bbox: Tuple[float, float, float, float],
        target_dist: float = _SPIGA_TARGET_DIST,
        crop_size: int = _SPIGA_CROP_SIZE,
    ) -> NDArray[np.float32]:
        """
        Forward affine matrix mapping image_bgr pixel coords to the
        crop_size x crop_size crop coords, matching dynaface's
        TargetCropAug(crop_size, target_dist): pad the bbox to a square of
        side max(w, h) * target_dist, centered on the bbox, then scale that
        square to crop_size.
        """
        x, y, w, h = bbox
        side = max(w, h) * target_dist
        cx, cy = x + w / 2.0, y + h / 2.0
        scale = crop_size / side
        center = crop_size / 2.0
        return np.array(
            [
                [scale, 0.0, center - scale * cx],
                [0.0, scale, center - scale * cy],
            ],
            dtype=np.float32,
        )

    @classmethod
    def _preprocess_landmarks(
        cls, image_bgr: NDArray[np.uint8], bbox: Tuple[float, float, float, float]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        affine = cls._crop_affine(bbox)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Matches PIL's Image.transform(..., Image.AFFINE) default behavior
        # (nearest-neighbor resample, zero-fill outside the source image),
        # which is what SPIGA's own pretreatment pipeline uses at crop time.
        crop = cv2.warpAffine(
            rgb,
            affine,
            (_SPIGA_CROP_SIZE, _SPIGA_CROP_SIZE),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        tensor = crop.astype(np.float32) / 255.0
        return tensor.transpose(2, 0, 1)[None], affine  # [1, 3, 256, 256] CHW

    # -- background removal -------------------------------------------------------

    def remove_background(self, image_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Run U^2-Net saliency segmentation and return a BGRA image with the
        background made transparent.
        """
        h, w = image_bgr.shape[:2]
        tensor = self._preprocess_background(image_bgr)

        # u2net.onnx has 7 outputs (d0..d6, multi-scale side outputs from
        # training); the first ("_bg_output_name", d0) is the final merged
        # saliency map and the one meant to be used as the mask.
        mask = self._background_session.run(
            [self._bg_output_name], {self._bg_input_name: tensor}
        )[0][
            0, 0
        ]  # [320, 320], sigmoid saliency map

        mask = cv2.resize(mask, (w, h))
        alpha = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)

        bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        return cast(NDArray[np.uint8], bgra)

    @staticmethod
    def _preprocess_background(image_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (320, 320)).astype(np.float32) / 255.0
        # U^2-Net was trained with ImageNet normalization stats.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        return rgb.transpose(2, 0, 1)[None].astype(np.float32)
