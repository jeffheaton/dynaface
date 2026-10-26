"""Constants shared across Dynaface: crop geometry, landmark indices and poses."""

from enum import Enum

STYLEGAN_WIDTH = 1024
STYLEGAN_LEFT_PUPIL = (640, 480)
STYLEGAN_RIGHT_PUPIL = (380, 480)
STYLEGAN_PUPIL_DIST = STYLEGAN_LEFT_PUPIL[0] - STYLEGAN_RIGHT_PUPIL[0]

STD_PUPIL_DIST = 63
DEFAULT_TILT_THRESHOLD = -1

LM_LEFT_PUPIL = 97
LM_RIGHT_PUPIL = 96

# Changed FILL_COLOR to a tuple to match expected type in safe_clip.
FILL_COLOR = (255, 255, 255)


class Pose(Enum):
    """
    The pose of a face: which way it faces the camera.

    Pass DETECT to AnalyzeFace.load_image() to have the pose classified as frontal
    or lateral, or one of the others to force it.
    """

    FRONTAL = "frontal"
    LATERAL = "lateral"
    QUARTER = "quarter"
    DETECT = "detect"
