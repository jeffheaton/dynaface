import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from facial_analysis import lateral, util, facial

logger = logging.getLogger(__name__)


def filter_measurements(data, items):
    return {item.name: data.get(item.name, None) for item in items}


def all_measures():
    return [
        AnalyzeFAI(),
        AnalyzeOralCommissureExcursion(),
        AnalyzeBrows(),
        AnalyzeDentalArea(),
        AnalyzeEyeArea(),
        AnalyzeIntercanthalDistance(),
        AnalyzeMouthLength(),
        AnalyzeNasalWidth(),
        AnalyzeOuterEyeCorners(),
        AnalyzeLateral(),
        AnalyzePosition(),
    ]


def to_degrees(r):
    # Convert the angle from radians to degrees
    tilt = r * (180 / math.pi)

    # Adjust the angle to be in a more intuitive range:
    # If the angle is greater than 90 degrees, subtract 180 degrees
    if tilt > 90:
        tilt -= 180
    # If the angle is less than -90 degrees, add 180 degrees
    elif tilt < -90:
        tilt += 180

    # Return the adjusted angle
    return tilt


class MeasureItem:
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.is_lateral = False
        self.is_frontal = False

    def __str__(self):
        return f"(name={self.name},enabled={self.enabled})"


class MeasureBase:
    def __init__(self):
        self.enabled = True
        self.items = []
        self.is_lateral = False
        self.is_frontal = False

    def update_for_type(self, lateral):
        for item in self.items:
            item.enabled = self.is_lateral if lateral else self.is_frontal

    def set_item_enabled(self, name, enabled):
        for item in self.items:
            if item.name == name:
                item.enabled = enabled

    def is_enabled(self, name):
        for item in self.items:
            if item.name == name:
                return item.enabled
        return True

    def set_enabled(self, enabled: bool):
        """
        Enable or disable the MeasureBase derived class and its items.

        Args:
            enabled (bool): True to enable, False to disable.
        """
        self.enabled = enabled
        for item in self.items:
            item.enabled = enabled

    def sync_items(self):
        for item in self.items:
            item.is_lateral = self.is_lateral
            item.is_frontal = self.is_frontal


class AnalyzeFAI(MeasureBase):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("fai")]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self):
        return "FAI"

    def calc(self, face, render=True):
        render2 = self.is_enabled("fai")
        d1 = face.measure(
            face.landmarks[64], face.landmarks[76], render=(render & render2), dir="l"
        )
        d2 = face.measure(
            face.landmarks[68], face.landmarks[82], render=(render & render2), dir="r"
        )
        if d1 > d2:
            fai = d1 - d2
        else:
            fai = d2 - d1

        if render & render2:
            txt = f"FAI={fai:.2f}"
            pos = face.analyze_next_pt(txt)
            face.write_text(pos, txt)
        return filter_measurements({"fai": fai}, (self.items))


class AnalyzeOralCommissureExcursion(MeasureBase):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("oce.l"), MeasureItem("oce.r")]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self):
        return "Oral CE"

    def calc(self, face, render=True):
        render2_l = self.is_enabled("oce.l")
        render2_r = self.is_enabled("oce.r")
        oce_r = face.measure(
            face.landmarks[76], face.landmarks[85], render=(render & render2_r), dir="l"
        )
        oce_l = face.measure(
            face.landmarks[82], face.landmarks[85], render=(render & render2_l), dir="r"
        )
        return filter_measurements({"oce.l": oce_l, "oce.r": oce_r}, self.items)


class AnalyzeBrows(MeasureBase):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("brow.d")]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self):
        return "Brow"

    def calc(self, face, render=True):
        render2 = self.is_enabled("brow.d")

        p = facial.util_get_pupils(face.landmarks)
        tilt = util.normalize_angle(util.calculate_face_rotation(p))

        if render & render2:
            right_brow = util.line_to_edge(
                img_size=1024, start_point=face.landmarks[35], angle=tilt
            )
            if not right_brow:
                return None
            face.arrow(face.landmarks[36], right_brow, apt2=False)

        # Diff
        diff = 0

        if render & render2:
            left_brow = util.line_to_edge(
                img_size=1024, start_point=face.landmarks[44], angle=tilt
            )
            if not left_brow:
                return None
            face.arrow(face.landmarks[44], left_brow, apt2=False)
            diff = abs(left_brow[1] - right_brow[1]) * face.pix2mm
            txt = f"d.brow={diff:.2f} mm"
            m = face.calc_text_size(txt)

            face.write_text(
                (face.width - (m[0][0] + 5), min(left_brow[1], right_brow[1]) - 10),
                txt,
            )

        return filter_measurements({"brow.d": diff}, self.items)


class AnalyzeDentalArea(MeasureBase):
    def __init__(self) -> None:
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

    def abbrev(self):
        return "Dental Display"

    def calc(self, face, render=True):
        render2_area = self.is_enabled("dental_area")
        render2_left = self.is_enabled("dental_left")
        render2_right = self.is_enabled("dental_right")
        render2_ratio = self.is_enabled("dental_ratio")
        render2_diff = self.is_enabled("dental_diff")

        contours_area = [
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

        contours_area = np.array(contours_area)

        try:
            contours_area_left, contours_area_right = util.split_polygon(
                contours_area, [p1, p2]
            )

            contours_area_left = np.array(contours_area_left, dtype=int)
            contours_area_right = np.array(contours_area_right, dtype=int)

            dental_area_right = face.measure_polygon(
                contours_area_right,
                face.pix2mm,
                render=(render & render2_right),
                color=(255, 0, 0),
            )

            dental_area_left = face.measure_polygon(
                contours_area_left,
                face.pix2mm,
                render=(render & render2_left),
                color=(0, 0, 255),
            )

            dental_area = dental_area_right + dental_area_left

            dental_ratio = util.symmetry_ratio(dental_area_left, dental_area_right)
            dental_diff = abs(dental_area_left - dental_area_right)
        except Exception as e:
            logger.error(f"Error in AnalyzeDentalArea.calc(): {e}")
            dental_area = dental_area_left = dental_area_right = 0
            dental_ratio = 1
            dental_diff = 0

        if render & render2_area:
            txt = f"dental={round(dental_area,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render & render2_left:
            txt = f"dental.left={round(dental_area_left,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render & render2_right:
            txt = f"dental.right={round(dental_area_right,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render & render2_ratio:
            txt = f"dental.ratio={round(dental_ratio,2)}"
            pos = face.analyze_next_pt(txt)
            face.write_text(pos, txt)

        if render & render2_diff:
            txt = f"dental.diff={round(dental_diff,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        return filter_measurements(
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
    def __init__(self) -> None:
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

    def abbrev(self):
        return "Eye Area"

    def calc(self, face, render=True):
        render2_eye_l = self.is_enabled("eye.left")
        render2_eye_r = self.is_enabled("eye.right")
        render2_eye_diff = self.is_enabled("eye.diff")
        render2_eye_ratio = self.is_enabled("eye.ratio")

        right_eye_area = face.measure_polygon(
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
            render=(render & render2_eye_r),
        )

        left_eye_area = face.measure_polygon(
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
            render=(render & render2_eye_l),
        )

        eye_area_diff = round(abs(right_eye_area - left_eye_area), 2)
        eye_area_ratio = util.symmetry_ratio(right_eye_area, left_eye_area)

        if render & render2_eye_r:
            face.write_text_sq(
                (face.landmarks[66][0] - 150, face.landmarks[66][1] + 20),
                f"R={round(right_eye_area,2)} mm",
            )

        if render & render2_eye_l:
            face.write_text_sq(
                (face.landmarks[74][0] - 50, face.landmarks[74][1] + 20),
                f"L={round(left_eye_area,2)} mm",
            )

        if render & render2_eye_diff:
            txt = f"eye.diff={round(eye_area_diff,2)} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render & render2_eye_ratio:
            txt = f"eye.ratio={round(eye_area_ratio,2)}"
            pos = face.analyze_next_pt(txt)
            face.write_text(pos, txt)

        return filter_measurements(
            {
                "eye.left": left_eye_area,
                "eye.right": right_eye_area,
                "eye.diff": eye_area_diff,
                "eye.ratio": eye_area_ratio,
            },
            self.items,
        )


class AnalyzePosition(MeasureBase):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("tilt"), MeasureItem("px2mm"), MeasureItem("pd")]
        self.is_frontal = True
        self.is_lateral = True
        self.sync_items()

    def abbrev(self):
        return "Position"

    def calc(self, face, render=True):
        render2_tilt = self.is_enabled("tilt")
        render2_px2mm = self.is_enabled("px2mm")
        render2_pd = self.is_enabled("pd")

        p = facial.util_get_pupils(face.landmarks)
        tilt = 0

        if p:
            landmarks = face.landmarks
            if render & render2_tilt:
                tilt = to_degrees(util.calculate_face_rotation(p))
                if face.face_rotation:
                    orig = to_degrees(face.face_rotation)
                    txt = f"tilt={round(orig,2)} -> {round(tilt,2)}"
                else:
                    txt = f"tilt={round(tilt,2)}"
                pos = face.analyze_next_pt(txt)
                face.write_text_sq(pos, txt, mark="o", up=15)
                p1, p2 = face.calc_bisect()
                face.line(p1, p2)

            if face.lateral:
                pd, pix2mm = 260, 0.24
            else:
                pd, pix2mm = facial.util_calc_pd(facial.util_get_pupils(landmarks))

            if render & render2_pd:
                txt = f"pd={round(pd,2)} px"
                pos = face.analyze_next_pt(txt)
                face.write_text(pos, txt)

            if render & render2_px2mm:
                txt = f"px2mm={round(pix2mm,2)}"
                pos = face.analyze_next_pt(txt)
                face.write_text(pos, txt)

        return filter_measurements(
            {"tilt": tilt, "px2mm": pix2mm, "pd": pd}, self.items
        )


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
        """
        return "Intercanthal Distance"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates the intercanthal distance (ID) using the landmarks at indices 88 and 92.

        Parameters:
        - `face`: A face object containing landmarks.
        - `render`: Whether to render the measurement visually.

        Returns:
        - A dictionary containing the intercanthal distance measurement.
        """
        render1: bool = self.is_enabled("id")

        # Measure the distance between landmarks 64 and 68
        d1: float = face.measure(
            face.landmarks[64], face.landmarks[68], render=(render and render1), dir="r"
        )

        # Return the filtered measurement result
        return filter_measurements({"id": d1}, self.items)


class AnalyzeMouthLength(MeasureBase):
    """
    Class to analyze ML (Mouth Length), the horizontal distance between the corners of the mouth
    """

    def __init__(self) -> None:
        """
        Initializes the measurement with default settings.
        """
        self.enabled: bool = True
        self.items: list[MeasureItem] = [MeasureItem("ml")]
        self.is_frontal: bool = True
        self.is_lateral: bool = False
        self.sync_items()

    def abbrev(self) -> str:
        """
        Returns the abbreviation of the measurement.
        """
        return "Mouth Length"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates the mouth length (ML) using the landmarks at indices 88 and 92.

        Parameters:
        - `face`: A face object containing landmarks.
        - `render`: Whether to render the measurement visually.

        Returns:
        - A dictionary containing the intercanthal distance measurement.
        """
        render1: bool = self.is_enabled("ml")

        # Measure the distance between landmarks 88 and 92
        d1: float = face.measure(
            face.landmarks[88], face.landmarks[92], render=(render and render1), dir="r"
        )

        # Return the filtered measurement result
        return filter_measurements({"ml": d1}, self.items)


class AnalyzeNasalWidth(MeasureBase):
    """
    Class to analyze NW (Nasal Width), the horizontal width of the nose at its widest point.
    """

    def __init__(self) -> None:
        """
        Initializes the measurement with default settings.
        """
        self.enabled: bool = True
        self.items: list[MeasureItem] = [MeasureItem("nw")]
        self.is_frontal: bool = True
        self.is_lateral: bool = False
        self.sync_items()

    def abbrev(self) -> str:
        """
        Returns the abbreviation of the measurement.
        """
        return "Nasal Width"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates the nasal width.

        Parameters:
        - `face`: A face object containing landmarks.
        - `render`: Whether to render the measurement visually.

        Returns:
        - A dictionary containing the intercanthal distance measurement.
        """
        render1: bool = self.is_enabled("nw")

        # Measure the distance between landmarks 55 and 59
        d1: float = face.measure(
            face.landmarks[55], face.landmarks[59], render=(render and render1), dir="a"
        )

        # Return the filtered measurement result
        return filter_measurements({"nw": d1}, self.items)


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
        self.items: list[MeasureItem] = [MeasureItem("oe")]
        self.is_frontal: bool = True
        self.is_lateral: bool = False
        self.sync_items()

    def abbrev(self) -> str:
        """
        Returns the abbreviation of the measurement.
        """
        return "Outer Eye Corners"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates the OE.

        Parameters:
        - `face`: A face object containing landmarks.
        - `render`: Whether to render the measurement visually.

        Returns:
        - A dictionary containing the intercanthal distance measurement.
        """
        render1: bool = self.is_enabled("oe")

        # Measure the distance between landmarks 60 and 72
        d1: float = face.measure(
            face.landmarks[60], face.landmarks[72], render=(render and render1), dir="r"
        )

        # Return the filtered measurement result
        return filter_measurements({"oe": d1}, self.items)


class AnalyzeLateral(MeasureBase):
    """
    Analyze several measurements in lateral view.
    NN distance soft tissue nasion to subnasal point
    NM distance subnasal point to mentolabial point
    NP distance subnasal point to soft tissue pogonion
    """

    def __init__(self) -> None:
        """
        Initializes the measurement with default settings.
        """
        super().__init__()
        self.enabled = True
        self.items = [
            MeasureItem("nn"),
            MeasureItem("nm"),
            MeasureItem("np"),
        ]
        self.is_frontal = False
        self.is_lateral = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Returns the abbreviation of the measurement.
        """
        return "Lateral Measures"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates lateral measures.

        Parameters:
        - `face`: A face object containing landmarks.
        - `render`: Whether to render the measurement visually.

        Returns:
        - A dictionary containing the lateral measurements.
        """
        render_nn = self.is_enabled("nn")
        render_nm = self.is_enabled("nm")
        render_np = self.is_enabled("np")

        if not face.lateral:
            return {}

        landmarks = face.lateral_landmarks

        # Measure the distance between landmarks for NN, NM, and NP
        nn = face.measure_curve(
            landmarks[lateral.LATERAL_LM_SOFT_TISSUE_NASION],
            landmarks[lateral.LATERAL_LM_SUBNASAL_POINT],
            face.sagittal_x,
            face.sagittal_y,
            render=(render and render_nn),
            dir="r",
        )
        nm = face.measure_curve(
            landmarks[lateral.LATERAL_LM_SUBNASAL_POINT],
            landmarks[lateral.LATERAL_LM_MENTO_LABIAL_POINT],
            face.sagittal_x,
            face.sagittal_y,
            render=(render and render_nm),
            dir="r",
        )
        np = face.measure_curve(
            landmarks[lateral.LATERAL_LM_SUBNASAL_POINT],
            landmarks[lateral.LATERAL_LM_SOFT_TISSUE_POGONION],
            face.sagittal_x,
            face.sagittal_y,
            render=(render and render_np),
            dir="r",
        )

        # Add text output for each measure
        if render and render_nn:
            txt_nn = f"NN={nn:.2f} mm"
            pos_nn = face.analyze_next_pt(txt_nn)
            face.write_text(pos_nn, txt_nn)

        if render and render_nm:
            txt_nm = f"NM={nm:.2f} mm"
            pos_nm = face.analyze_next_pt(txt_nm)
            face.write_text(pos_nm, txt_nm)

        if render and render_np:
            txt_np = f"NP={np:.2f} mm"
            pos_np = face.analyze_next_pt(txt_np)
            face.write_text(pos_np, txt_np)

        # Return the filtered measurement result
        return filter_measurements({"nn": nn, "nm": nm, "np": np}, self.items)


# ID intercanthal distance
# ML mouth length
# NW nasal width
# OE distance between outer eye corners
# NN distance soft tissue nasion to subnasal point
# NM distance subnasal point to mentolabial point
# NP distance subnasal point to soft tissue pogonion
