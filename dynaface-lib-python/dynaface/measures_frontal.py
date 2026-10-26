"""Measures of a frontal (face-on) view."""

import logging
from typing import Any, Dict, List

import numpy as np

from dynaface import facial, util
from dynaface.measures_base import MeasureBase, MeasureItem, select_outputs

logger = logging.getLogger(__name__)


class AnalyzeFAI(MeasureBase):
    """
    Facial Asymmetry Index (FAI): how much the distance from each eye's inner
    corner to the same side's mouth corner differs between the two sides, in
    millimeters.

    Output: fai.
    """

    def __init__(self) -> None:
        """
        Initialize the AnalyzeFAI measurement.
        """
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("fai")]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the FAI measurement.

        Returns:
            str: Abbreviation string.
        """
        return "FAI"

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        """
        Calculate the Facial Asymmetry Index (FAI).

        Args:
            face (Any): Face object containing landmarks and measurement methods.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, Any]: fai, in millimeters.
        """
        render2: bool = self.is_enabled("fai")
        d1: float = face.measure(
            face.landmarks[64], face.landmarks[76], render=(render and render2), dir="l"
        )
        d2: float = face.measure(
            face.landmarks[68], face.landmarks[82], render=(render and render2), dir="r"
        )
        fai: float = d1 - d2 if d1 > d2 else d2 - d1

        if render and render2:
            txt = f"FAI={fai:.2f}"
            pos = face.analyze_next_pt(txt)
            face.write_text(pos, txt)
        return select_outputs({"fai": fai}, self.items)


class AnalyzeOralCommissureExcursion(MeasureBase):
    """
    Analyze the Oral Commissure Excursion measurement.
    """

    def __init__(self) -> None:
        """
        Initialize the AnalyzeOralCommissureExcursion measurement.
        """
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("oce.l"), MeasureItem("oce.r")]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the Oral Commissure Excursion measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Oral CE"

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        """
        Calculate the Oral Commissure Excursion measurement.

        Args:
            face (Any): Face object containing landmarks and measurement methods.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, Any]: Filtered measurement results.
        """
        render2_l: bool = self.is_enabled("oce.l")
        render2_r: bool = self.is_enabled("oce.r")
        oce_r: float = face.measure(
            face.landmarks[76],
            face.landmarks[85],
            render=(render and render2_r),
            dir="l",
        )
        oce_l: float = face.measure(
            face.landmarks[82],
            face.landmarks[85],
            render=(render and render2_l),
            dir="r",
        )
        return select_outputs({"oce.l": oce_l, "oce.r": oce_r}, self.items)


class AnalyzeBrows(MeasureBase):
    """
    Analyze the brow distance measurement.
    """

    def __init__(self) -> None:
        """
        Initialize the AnalyzeBrows measurement.
        """
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("brow.d")]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the brow measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Brow"

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        """
        Calculate the brow distance measurement.

        Args:
            face (Any): Face object containing landmarks and measurement methods.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, Any]: Filtered measurement results.
        """
        render2: bool = self.is_enabled("brow.d")

        p: Any = facial.util_get_pupils(face.landmarks)
        tilt: float = util.normalize_angle(util.calculate_face_rotation(p))

        right_brow: Any = util.line_to_edge(
            img_size=1024, start_point=face.landmarks[35], angle=tilt
        )
        if not right_brow:
            return {}

        if render and render2:
            face.arrow(face.landmarks[36], right_brow, apt2=False)

        diff: float = 0

        left_brow: Any = util.line_to_edge(
            img_size=1024, start_point=face.landmarks[44], angle=tilt
        )

        if not left_brow:
            return {}

        diff = abs(left_brow[1] - right_brow[1]) * face.pix2mm
        txt = f"d.brow={diff:.2f} mm"
        m: Any = face.calc_text_size(txt)

        if render and render2:
            face.arrow(face.landmarks[44], left_brow, apt2=False)
            face.write_text(
                (face.width - (m[0][0] + 5), min(left_brow[1], right_brow[1]) - 10),
                txt,
            )

        return select_outputs({"brow.d": diff}, self.items)


DENTAL_MIN_AREA_MM2 = 1e-6


class AnalyzeDentalArea(MeasureBase):
    """
    Analyze the dental area measurements.
    """

    def __init__(self) -> None:
        """
        Initialize the AnalyzeDentalArea measurement.
        """
        super().__init__()
        self.enabled = True
        self.items = [
            MeasureItem("dental_area"),
            MeasureItem("dental_left"),
            MeasureItem("dental_right"),
            MeasureItem("dental_ratio"),
            MeasureItem("dental_diff"),
        ]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the dental area measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Dental Display"

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        """
        Calculate the dental area measurements.

        Args:
            face (Any): Face object containing landmarks and measurement methods.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, Any]: Filtered measurement results.
        """
        render2_area: bool = self.is_enabled("dental_area")
        render2_left: bool = self.is_enabled("dental_left")
        render2_right: bool = self.is_enabled("dental_right")
        render2_ratio: bool = self.is_enabled("dental_ratio")
        render2_diff: bool = self.is_enabled("dental_diff")

        # Build a list of contour landmarks then convert to an ndarray.
        contours_area_list = [
            face.landmarks[88],
            face.landmarks[89],
            face.landmarks[90],
            face.landmarks[91],
            face.landmarks[92],
            face.landmarks[93],
            face.landmarks[94],
            face.landmarks[95],
        ]
        p1, p2 = face.calc_bisect()
        contours_area = np.array(contours_area_list)

        def half_area(image_left: bool, enabled: bool, color: Any) -> float:
            # Clipping (rather than util.split_polygon) copes with a closed mouth,
            # whose inner-lip outline has no area: the halves just measure ~0.
            half = util.clip_polygon_to_side(contours_area, (p1, p2), image_left)
            if len(half) < 3:
                return 0.0
            area = float(
                face.measure_polygon(
                    half,
                    face.pix2mm,
                    render=(render and enabled),
                    color=color,
                )
            )
            # Floating-point dust from a closed mouth counts as zero, so the
            # symmetry ratio of two empty halves stays 1.
            return area if area >= DENTAL_MIN_AREA_MM2 else 0.0

        try:
            # Sides are the person's, like eye.right and oce.r: their right is the
            # image-left half.
            dental_area_right: float = half_area(True, render2_right, (255, 0, 0))
            dental_area_left: float = half_area(False, render2_left, (0, 0, 255))

            dental_area: float = dental_area_right + dental_area_left

            dental_ratio: float = util.symmetry_ratio(
                dental_area_left, dental_area_right
            )
            dental_diff: float = abs(dental_area_left - dental_area_right)
        except Exception as e:
            logger.error(f"Error in AnalyzeDentalArea.calc(): {e}")
            dental_area = dental_area_left = dental_area_right = 0
            dental_ratio = 1
            dental_diff = 0

        if render and render2_area:
            txt = f"dental={round(dental_area,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render and render2_left:
            txt = f"dental.left={round(dental_area_left,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render and render2_right:
            txt = f"dental.right={round(dental_area_right,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render and render2_ratio:
            txt = f"dental.ratio={round(dental_ratio,2)}"
            pos = face.analyze_next_pt(txt)
            face.write_text(pos, txt)

        if render and render2_diff:
            txt = f"dental.diff={round(dental_diff,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        return select_outputs(
            {
                "dental_area": dental_area,
                "dental_left": dental_area_left,
                "dental_right": dental_area_right,
                "dental_ratio": dental_ratio,
                "dental_diff": dental_diff,
            },
            self.items,
        )


class AnalyzeEyeArea(MeasureBase):
    """
    Analyze the eye area measurements.
    """

    def __init__(self) -> None:
        """
        Initialize the AnalyzeEyeArea measurement.
        """
        super().__init__()
        self.enabled = True
        self.items = [
            MeasureItem("eye.left"),
            MeasureItem("eye.right"),
            MeasureItem("eye.diff"),
            MeasureItem("eye.ratio"),
        ]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the eye area measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Eye Area"

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        """
        Calculate the eye area measurements.

        Args:
            face (Any): Face object containing landmarks and measurement methods.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, Any]: Filtered measurement results.
        """
        render2_eye_l: bool = self.is_enabled("eye.left")
        render2_eye_r: bool = self.is_enabled("eye.right")
        render2_eye_diff: bool = self.is_enabled("eye.diff")
        render2_eye_ratio: bool = self.is_enabled("eye.ratio")

        right_eye_area: float = face.measure_polygon(
            [
                face.landmarks[60],
                face.landmarks[61],
                face.landmarks[62],
                face.landmarks[63],
                face.landmarks[64],
                face.landmarks[65],
                face.landmarks[66],
                face.landmarks[67],
            ],
            face.pix2mm,
            render=(render and render2_eye_r),
        )

        left_eye_area: float = face.measure_polygon(
            [
                face.landmarks[68],
                face.landmarks[69],
                face.landmarks[70],
                face.landmarks[71],
                face.landmarks[72],
                face.landmarks[73],
                face.landmarks[74],
                face.landmarks[75],
            ],
            face.pix2mm,
            render=(render and render2_eye_l),
        )

        eye_area_diff: float = round(abs(right_eye_area - left_eye_area), 2)
        eye_area_ratio: float = util.symmetry_ratio(right_eye_area, left_eye_area)

        if render and render2_eye_r:
            face.write_text_sq(
                (face.landmarks[66][0] - 150, face.landmarks[66][1] + 20),
                f"R={round(right_eye_area,2)} mm",
            )

        if render and render2_eye_l:
            face.write_text_sq(
                (face.landmarks[74][0] - 50, face.landmarks[74][1] + 20),
                f"L={round(left_eye_area,2)} mm",
            )

        if render and render2_eye_diff:
            txt = f"eye.diff={round(eye_area_diff,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render and render2_eye_ratio:
            txt = f"eye.ratio={round(eye_area_ratio,2)}"
            pos = face.analyze_next_pt(txt)
            face.write_text(pos, txt)

        return select_outputs(
            {
                "eye.left": left_eye_area,
                "eye.right": right_eye_area,
                "eye.diff": eye_area_diff,
                "eye.ratio": eye_area_ratio,
            },
            self.items,
        )


class AnalyzePosition(MeasureBase):
    """
    Analyze facial position measurements including tilt, pixel-to-millimeter conversion, and pupillary distance.
    """

    def __init__(self) -> None:
        """
        Initialize the AnalyzePosition measurement.
        """
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("tilt"), MeasureItem("px2mm"), MeasureItem("pd")]
        self.is_frontal = True
        self.is_lateral = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the position measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Position"

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        """
        Calculate the position measurements (tilt, px2mm, pd).

        Args:
            face (Any): Face object containing landmarks and measurement methods.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, Any]: Filtered measurement results.
        """
        render2_tilt: bool = self.is_enabled("tilt")
        render2_px2mm: bool = self.is_enabled("px2mm")
        render2_pd: bool = self.is_enabled("pd")

        p: Any = facial.util_get_pupils(face.landmarks)
        tilt: float = 0.0
        pd: float = 260
        pix2mm: float = 0.24

        if p:
            landmarks: Any = face.landmarks
            tilt = util.to_degrees(util.calculate_face_rotation(p))
            if render and render2_tilt:
                if face.face_rotation:
                    orig: float = util.to_degrees(face.face_rotation)
                    txt = f"tilt={round(orig,2)} -> {round(tilt,2)}"
                else:
                    txt = f"tilt={round(tilt,2)}"
                pos = face.analyze_next_pt(txt)
                face.write_text_sq(pos, txt, mark="o", up=15)
                p1, p2 = face.calc_bisect()
                face.line(p1, p2)

            if not face.lateral:
                pd, pix2mm = facial.util_calc_pd(facial.util_get_pupils(landmarks))
            else:
                pd, _ = facial.util_calc_pd(facial.util_get_pupils(landmarks))

            if render and render2_pd:
                txt = f"pd={round(pd,2)} px"
                pos = face.analyze_next_pt(txt)
                face.write_text(pos, txt)

            if render and render2_px2mm:
                txt = f"px2mm={round(pix2mm,2)}"
                pos = face.analyze_next_pt(txt)
                face.write_text(pos, txt)

        return select_outputs({"tilt": tilt, "px2mm": pix2mm, "pd": pd}, self.items)


class AnalyzePose(MeasureBase):
    """
    Analyze facial pose, pitch, yaw and roll. Pitch is when you nod your head up and down, like saying "yes." Yaw is
    when you turn your head left and right, like saying "no." Roll is when you tilt your head sideways toward
    your shoulder, like trying to touch your ear to your shoulder.
    """

    def __init__(self) -> None:
        """
        Initialize the AnalyzePose measurement.
        """
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("pitch"), MeasureItem("roll"), MeasureItem("yaw")]
        self.is_frontal = True
        self.is_lateral = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the pose measurements.

        Returns:
            str: Abbreviation string.
        """
        return "Pose"

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        """
        Calculate the pose measurements (pitch, roll, yaw).

        Args:
            face (Any): Face object containing landmarks and measurement methods.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, Any]: Filtered measurement results.
        """
        render2_pitch: bool = self.is_enabled("pitch")
        render2_roll: bool = self.is_enabled("roll")
        render2_yaw: bool = self.is_enabled("yaw")

        # pitch = util.to_degrees(face.pitch)
        # roll = util.to_degrees(face.roll)
        # yaw = util.to_degrees(face.yaw)

        pitch = face.pitch
        roll = face.roll
        yaw = face.yaw

        if render and render2_pitch:
            txt = f"Pitch: {pitch:.2f}"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt, mark="o", up=15)

        if render and render2_roll:
            txt = f"Roll: {roll:.2f}"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt, mark="o", up=15)

        if render and render2_yaw:
            txt = f"Yaw: {yaw:.2f}"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt, mark="o", up=15)

        return select_outputs({"pitch": pitch, "roll": roll, "yaw": yaw}, self.items)


class AnalyzeIntercanthalDistance(MeasureBase):
    """
    Class to analyze intercanthal distance (ID), which is the distance between the inner corners of the eyes.
    """

    def __init__(self) -> None:
        """
        Initializes the measurement with default settings.
        """
        self.enabled: bool = True
        self.items: List[MeasureItem] = [MeasureItem("id")]
        self.is_frontal: bool = True
        self.is_lateral: bool = False
        self.sync_items()

    def abbrev(self) -> str:
        """
        Returns the abbreviation of the measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Intercanthal Distance"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates the intercanthal distance (ID) using the landmarks at indices 64 and 68.

        Parameters:
            face (Any): A face object containing landmarks.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, float]: A dictionary containing the intercanthal distance measurement.
        """
        render1: bool = self.is_enabled("id")
        d1: float = face.measure(
            face.landmarks[64], face.landmarks[68], render=(render and render1), dir="r"
        )
        return select_outputs({"id": d1}, self.items)


class AnalyzeMouthLength(MeasureBase):
    """
    Class to analyze ML (Mouth Length), the horizontal distance between the corners of the mouth.
    """

    def __init__(self) -> None:
        """
        Initializes the measurement with default settings.
        """
        self.enabled: bool = True
        self.items: List[MeasureItem] = [MeasureItem("ml")]
        self.is_frontal: bool = True
        self.is_lateral: bool = False
        self.sync_items()

    def abbrev(self) -> str:
        """
        Returns the abbreviation of the measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Mouth Length"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates the mouth length (ML) using the landmarks at indices 88 and 92.

        Parameters:
            face (Any): A face object containing landmarks.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, float]: A dictionary containing the mouth length measurement.
        """
        render1: bool = self.is_enabled("ml")
        d1: float = face.measure(
            face.landmarks[88], face.landmarks[92], render=(render and render1), dir="r"
        )
        return select_outputs({"ml": d1}, self.items)


class AnalyzeNoseFrontal(MeasureBase):
    """
    Frontal nose widths, in millimeters, between landmarks on the nose's lower outline:

    nose.alar: alar width, between the outer edges of the nostrils (WFLW 55 and 59).
    nose.nostril: between the points midway along each nostril (WFLW 56 and 58).

    These replace four earlier columns that held only these two widths: nostril was the
    alar span padded by 20%, dorsal.base the alar span, and nose.tip and dorsal.bridge
    both the 56-58 span. WFLW has no landmarks on the sides of the nose higher up, so
    widths along the dorsum can't be measured from landmarks.
    """

    def __init__(self) -> None:
        """
        Initializes the measurement with default settings.
        """
        self.enabled: bool = True
        self.items: List[MeasureItem] = [
            MeasureItem("nose.alar"),
            MeasureItem("nose.nostril"),
        ]
        self.is_frontal: bool = True
        self.is_lateral: bool = False
        self.sync_items()

    def abbrev(self) -> str:
        """
        Returns the abbreviation of the measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Nose Frontal"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates the alar and nostril widths.

        Parameters:
            face (Any): A face object containing landmarks.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, float]: nose.alar and nose.nostril in millimeters.
        """
        render_alar: bool = self.is_enabled("nose.alar")
        render_nostril: bool = self.is_enabled("nose.nostril")

        alar: float = face.measure(
            face.landmarks[55],
            face.landmarks[59],
            render=(render and render_alar),
            dir="s",
        )
        nostril: float = face.measure(
            face.landmarks[56],
            face.landmarks[58],
            render=(render and render_nostril),
            dir="s",
        )

        for enabled, txt in (
            (render_alar, f"nose.alar={alar:.2f}"),
            (render_nostril, f"nose.nostril={nostril:.2f}"),
        ):
            if render and enabled:
                pos = face.analyze_next_pt(txt)
                face.write_text(pos, txt)

        return select_outputs({"nose.alar": alar, "nose.nostril": nostril}, self.items)


class AnalyzeOuterEyeCorners(MeasureBase):
    """
    OE (Distance Between Outer Eye Corners): The horizontal distance between the outermost points of
    the eyes (lateral canthi).
    """

    def __init__(self) -> None:
        """
        Initializes the measurement with default settings.
        """
        self.enabled: bool = True
        self.items: List[MeasureItem] = [MeasureItem("oe")]
        self.is_frontal: bool = True
        self.is_lateral: bool = False
        self.sync_items()

    def abbrev(self) -> str:
        """
        Returns the abbreviation of the measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Outer Eye Corners"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates the outer eye corners distance.

        Parameters:
            face (Any): A face object containing landmarks.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, float]: A dictionary containing the outer eye corners measurement.
        """
        render1: bool = self.is_enabled("oe")
        d1: float = face.measure(
            face.landmarks[60], face.landmarks[72], render=(render and render1), dir="r"
        )
        return select_outputs({"oe": d1}, self.items)


class AnalyzeLandmarks(MeasureBase):
    """Output all facial landmark (x, y) coordinates as named CSV columns."""

    # Number of SPIGA wflw landmarks
    NUM_LANDMARKS = 98

    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.items = [
            item
            for i in range(1, self.NUM_LANDMARKS + 1)
            for item in (
                MeasureItem(f"landmark-{i}-x"),
                MeasureItem(f"landmark-{i}-y"),
            )
        ]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the landmarks measure.

        Returns:
            str: Abbreviation string.
        """
        return "Landmarks"

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        """
        Return the position of all 98 landmarks, drawing them when render is True.

        Args:
            face (Any): Face object containing landmarks and measurement methods.
            render (bool): Whether to render the landmarks visually.

        Returns:
            Dict[str, Any]: landmark-1-x, landmark-1-y and so on through
                landmark-98-y, in pixels, numbered from 1.
        """
        if render:
            # Draw black outline (radius 3) then white fill (radius 2) so dots
            # are visible against any background color.
            face.draw_landmarks(size=1.0, color=(0, 0, 0))
            face.draw_landmarks(size=0.5, color=(255, 255, 255))

        data: Dict[str, Any] = {}
        for i, (x, y) in enumerate(face.landmarks):
            n = i + 1  # 1-indexed
            data[f"landmark-{n}-x"] = x
            data[f"landmark-{n}-y"] = y

        return select_outputs(data, self.items)
