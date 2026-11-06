"""
A model whose output no working model gives (values that aren't numbers, the same
value everywhere, or landmarks all at one point) raises ModelOutputError, so a device
that computes the models wrongly isn't mistaken for an image without a face. Mirrors
dynaface-lib-dotnet's ModelOutputTests.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import facial, models  # noqa: E402
from dynaface.const import Pose  # noqa: E402
from dynaface.dynaface_onnx import (  # noqa: E402
    DynafaceOnnxInference,
    ModelOutputError,
    _generate_blaze_face_anchors,
)

IMAGE = np.full((256, 256, 3), 128, dtype=np.uint8)
RNG = np.random.default_rng(0)
# Outputs like a working model's: scores that differ, none high enough for a face.
NO_FACE = (
    RNG.normal(0, 5, (1, 896, 16)).astype(np.float32),
    RNG.normal(-8, 2, (1, 896, 1)).astype(np.float32),
)
LANDMARKS = (
    (0.3 + 0.4 * RNG.random((1, 98, 2))).astype(np.float32),
    np.array([[5.0, -3.0, 2.0, 0.0, 0.0, 30.0]], dtype=np.float32),
)
MASK = RNG.random((1, 1, 320, 320)).astype(np.float32)


class _Session:
    """Stands in for an ONNX Runtime session, giving fixed outputs."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)

    def run(self, names, feeds):
        return self.outputs


def backend(face=NO_FACE, landmarks=LANDMARKS, mask=MASK):
    """A DynafaceOnnxInference whose models give these outputs."""
    inference = object.__new__(DynafaceOnnxInference)
    inference._face_session = _Session(*face)
    inference._face_input_name = "input"
    inference._face_output_names = ["regressors", "classificators"]
    inference._face_anchors = _generate_blaze_face_anchors()
    inference._landmark_session = _Session(*landmarks)
    inference._model3d = None
    inference._cam_matrix = None
    inference._background_session = _Session(mask)
    inference._bg_input_name = "input"
    inference._bg_output_name = "output"
    return inference


def with_nan(output):
    output = output.copy()
    output.flat[output.size // 2] = np.nan
    return output


class TestModelOutputs(unittest.TestCase):
    def test_working_models(self):
        inference = backend()
        self.assertEqual(inference.detect_face(IMAGE), (None, None))  # no face
        landmarks, pose = inference.find_landmarks(IMAGE, (64, 64, 128, 128))
        self.assertEqual(landmarks.shape, (98, 2))
        self.assertEqual(inference.remove_background(IMAGE).shape, (256, 256, 4))

    def test_face_detector(self):
        regressors, scores = NO_FACE
        for name, face in (
            ("NaN score", (regressors, with_nan(scores))),
            ("infinite box", (np.full_like(regressors, np.inf), scores)),
            ("every score the same", (regressors, np.zeros_like(scores))),
        ):
            with self.subTest(name):
                with self.assertRaisesRegex(ModelOutputError, "face detector"):
                    backend(face=face).detect_face(IMAGE)

    def test_landmark_model(self):
        landmarks, pose = LANDMARKS
        for name, output in (
            ("NaN landmark", (with_nan(landmarks), pose)),
            ("NaN pose", (landmarks, with_nan(pose))),
            ("all at one point", (np.full_like(landmarks, 0.5), pose)),
            ("all zero", (np.zeros_like(landmarks), np.zeros_like(pose))),
        ):
            with self.subTest(name):
                with self.assertRaisesRegex(ModelOutputError, "landmark model"):
                    backend(landmarks=output).find_landmarks(IMAGE, (64, 64, 128, 128))

    def test_background_model(self):
        for name, mask in (
            ("NaN", with_nan(MASK)),
            ("every pixel the same", np.zeros_like(MASK)),
        ):
            with self.subTest(name):
                with self.assertRaisesRegex(ModelOutputError, "background model"):
                    backend(mask=mask).remove_background(IMAGE)

    def test_loading_an_image_reports_it(self):
        # Zeroed outputs, as from a GPU that runs the models but computes nothing:
        # they used to look like an image without a face.
        broken = backend(face=tuple(np.zeros_like(output) for output in NO_FACE))
        face = facial.AnalyzeFace(measures=[])
        with mock.patch.object(models, "onnx_model", broken), mock.patch.object(
            facial, "are_models_init", return_value=True
        ):
            with self.assertRaises(ModelOutputError) as raised:
                face.load_image(IMAGE, crop=True, force_pose=Pose.DETECT)
        self.assertIn("CPU", str(raised.exception))

    def test_models_module_has_it(self):
        self.assertIs(models.ModelOutputError, ModelOutputError)
        self.assertTrue(issubclass(ModelOutputError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
