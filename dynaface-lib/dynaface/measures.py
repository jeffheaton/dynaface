import logging
import math
from typing import Any, Dict, List

import numpy as np
from dynaface import lateral, util, facial

logger = logging.getLogger(__name__)


def filter_measurements(
    data: Dict[str, Any], items: List["MeasureItem"]
) -> Dict[str, Any]:
    """
    Filter measurements from a data dictionary based on the provided measurement items.

    Args:
        data (Dict[str, Any]): Dictionary containing measurement values.
        items (List[MeasureItem]): List of measurement items to filter.

    Returns:
        Dict[str, Any]: A dictionary with keys from measurement item names and their corresponding values from data.
    """
    return {item.name: data.get(item.name, None) for item in items}


def all_measures() -> List["MeasureBase"]:
    """
    Return a list of all measurement analysis objects.

    Returns:
        List[MeasureBase]: List of measurement analysis objects.
    """
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


def to_degrees(r: float) -> float:
    """
    Convert an angle from radians to degrees and adjust it to be in a more intuitive range.

    Args:
        r (float): Angle in radians.

    Returns:
        float: Adjusted angle in degrees.
    """
    tilt = r * (180 / math.pi)

    if tilt > 90:
        tilt -= 180
    elif tilt < -90:
        tilt += 180

    return tilt


class MeasureItem:
    """
    Represents an individual measurement item with a name and enabled state.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        """
        Initialize a MeasureItem.

        Args:
            name (str): The name of the measurement.
            enabled (bool): Whether the measurement is enabled.
        """
        self.name: str = name
        self.enabled: bool = enabled
        self.is_lateral: bool = False
        self.is_frontal: bool = False

    def __str__(self) -> str:
        """
        Return a string representation of the MeasureItem.

        Returns:
            str: String representation.
        """
        return f"(name={self.name},enabled={self.enabled})"


class MeasureBase:
    """
    Base class for measurement analysis objects.
    """

    def __init__(self) -> None:
        """
        Initialize the MeasureBase with default settings.
        """
        self.enabled: bool = True
        self.items: List[MeasureItem] = []
        self.is_lateral: bool = False
        self.is_frontal: bool = False

    def update_for_type(self, lateral: bool) -> None:
        """
        Update the enabled state of each measurement item based on view type.

        Args:
            lateral (bool): True if the measurement is lateral, False if frontal.
        """
        for item in self.items:
            item.enabled = self.is_lateral if lateral else self.is_frontal

    def set_item_enabled(self, name: str, enabled: bool) -> None:
        """
        Set the enabled state for a specific measurement item by name.

        Args:
            name (str): The name of the measurement item.
            enabled (bool): True to enable, False to disable.
        """
        for item in self.items:
            if item.name == name:
                item.enabled = enabled

    def is_enabled(self, name: str) -> bool:
        """
        Check if a specific measurement item is enabled.

        Args:
            name (str): The name of the measurement item.

        Returns:
            bool: True if the item is enabled, False otherwise.
        """
        for item in self.items:
            if item.name == name:
                return item.enabled
        return True

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the MeasureBase derived class and its items.

        Args:
            enabled (bool): True to enable, False to disable.
        """
        self.enabled = enabled
        for item in self.items:
            item.enabled = enabled

    def sync_items(self) -> None:
        """
        Synchronize each measurement item with the base class view settings.
        """
        for item in self.items:
            item.is_lateral = self.is_lateral
            item.is_frontal = self.is_frontal


class AnalyzeFAI(MeasureBase):
    """
    Analyze the Facial Angle of Incisor (FAI) measurement.
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
        Calculate the Facial Angle of Incisor (FAI) measurement.

        Args:
            face (Any): Face object containing landmarks and measurement methods.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, Any]: Filtered measurement results.
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
        return filter_measurements({"fai": fai}, self.items)


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
        return filter_measurements({"oce.l": oce_l, "oce.r": oce_r}, self.items)


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

        return filter_measurements({"brow.d": diff}, self.items)


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

        try:
            # Pass a tuple as the second argument.
            contours_area_left, contours_area_right = util.split_polygon(
                contours_area, (p1, p2)
            )

            contours_area_left = np.array(contours_area_left, dtype=int)
            contours_area_right = np.array(contours_area_right, dtype=int)

            dental_area_right: float = face.measure_polygon(
                contours_area_right,
                face.pix2mm,
                render=(render and render2_right),
                color=(255, 0, 0),
            )

            dental_area_left: float = face.measure_polygon(
                contours_area_left,
                face.pix2mm,
                render=(render and render2_left),
                color=(0, 0, 255),
            )

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
            if render and render2_tilt:
                tilt = to_degrees(util.calculate_face_rotation(p))
                if face.face_rotation:
                    orig: float = to_degrees(face.face_rotation)
                    txt = f"tilt={round(orig,2)} -> {round(tilt,2)}"
                else:
                    txt = f"tilt={round(tilt,2)}"
                pos = face.analyze_next_pt(txt)
                face.write_text_sq(pos, txt, mark="o", up=15)
                p1, p2 = face.calc_bisect()
                face.line(p1, p2)

            if not face.lateral:
                pd, pix2mm = facial.util_calc_pd(facial.util_get_pupils(landmarks))

            if render and render2_pd:
                txt = f"pd={round(pd,2)} px"
                pos = face.analyze_next_pt(txt)
                face.write_text(pos, txt)

            if render and render2_px2mm:
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
        return filter_measurements({"id": d1}, self.items)


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
        self.items: List[MeasureItem] = [MeasureItem("nw")]
        self.is_frontal: bool = True
        self.is_lateral: bool = False
        self.sync_items()

    def abbrev(self) -> str:
        """
        Returns the abbreviation of the measurement.

        Returns:
            str: Abbreviation string.
        """
        return "Nasal Width"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculates the nasal width.

        Parameters:
            face (Any): A face object containing landmarks.
            render (bool): Whether to render the measurement visually.

        Returns:
            Dict[str, float]: A dictionary containing the nasal width measurement.
        """
        render1: bool = self.is_enabled("nw")
        d1: float = face.measure(
            face.landmarks[55], face.landmarks[59], render=(render and render1), dir="a"
        )
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
        return filter_measurements({"oe": d1}, self.items)


class AnalyzeLateral(MeasureBase):
    """
    Analyze several measurements in lateral view.
    NN: Distance from soft tissue nasion to subnasal point.
    NM: Distance from subnasal point to mentolabial point.
    NP: Distance from subnasal point to soft tissue pogonion.
    """

    def __init__(self) -> None:
        """
        Initializes the lateral measurements with default settings.
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
        Returns the abbreviation for the lateral measurements.

        Returns:
            str: Abbreviation string.
        """
        return "Lateral Measures"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculate the lateral measurements.

        Args:
            face (Any): A face object containing lateral landmarks and measurement methods.
            render (bool): Whether to render the measurements visually.

        Returns:
            Dict[str, float]: Filtered measurement results for NN, NM, and NP.
        """
        render_nn: bool = self.is_enabled("nn")
        render_nm: bool = self.is_enabled("nm")
        render_np: bool = self.is_enabled("np")

        if not face.lateral:
            return {}

        landmarks: Any = face.lateral_landmarks

        nn: float = face.measure_curve(
            landmarks[lateral.LATERAL_LM_SOFT_TISSUE_NASION],
            landmarks[lateral.LATERAL_LM_SUBNASAL_POINT],
            face.sagittal_x,
            face.sagittal_y,
            render=(render and render_nn),
            dir="r",
        )
        nm: float = face.measure_curve(
            landmarks[lateral.LATERAL_LM_SUBNASAL_POINT],
            landmarks[lateral.LATERAL_LM_MENTO_LABIAL_POINT],
            face.sagittal_x,
            face.sagittal_y,
            render=(render and render_nm),
            dir="r",
        )
        np_val: float = face.measure_curve(
            landmarks[lateral.LATERAL_LM_SUBNASAL_POINT],
            landmarks[lateral.LATERAL_LM_SOFT_TISSUE_POGONION],
            face.sagittal_x,
            face.sagittal_y,
            render=(render and render_np),
            dir="r",
        )

        if render and render_nn:
            txt_nn: str = f"NN={nn:.2f} mm"
            pos_nn: Any = face.analyze_next_pt(txt_nn)
            face.write_text(pos_nn, txt_nn)

        if render and render_nm:
            txt_nm: str = f"NM={nm:.2f} mm"
            pos_nm: Any = face.analyze_next_pt(txt_nm)
            face.write_text(pos_nm, txt_nm)

        if render and render_np:
            txt_np: str = f"NP={np_val:.2f} mm"
            pos_np: Any = face.analyze_next_pt(txt_np)
            face.write_text(pos_np, txt_np)

        return filter_measurements({"nn": nn, "nm": nm, "np": np_val}, self.items)
