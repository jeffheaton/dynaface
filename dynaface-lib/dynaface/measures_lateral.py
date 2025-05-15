import logging
from typing import Any, Dict
import numpy as np

from dynaface import lateral, util

logger = logging.getLogger(__name__)

from dynaface.measures_base import MeasureBase, MeasureItem, filter_measurements


class AnalyzeLateral(MeasureBase):
    """
    Analyze several measurements in lateral view.
    NN: Distance from soft tissue nasion to subnasal point.
    NM: Distance from subnasal point to mentolabial point.
    NP: Distance from subnasal point to soft tissue pogonion.
    NFA: Nasofrontal Angle at Soft Tissue Nasion formed by Glabella and Nasal Tip.
    NLA: Nasolabial Angle at Subnasal Point formed by Nasal Tip and Soft Tissue Pogonion.
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
            MeasureItem("nfa"),
            MeasureItem("nla"),
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
            Dict[str, float]: Filtered measurement results for NN, NM, NP, NFA, and NLA.
        """
        render_nn: bool = self.is_enabled("nn")
        render_nm: bool = self.is_enabled("nm")
        render_np: bool = self.is_enabled("np")
        render_nfa: bool = self.is_enabled("nfa")
        render_nla: bool = self.is_enabled("nla")

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

        def calculate_angle(pt_center, pt1, pt2):
            v1 = np.array(pt1) - np.array(pt_center)
            v2 = np.array(pt2) - np.array(pt_center)
            angle_rad = np.arccos(
                np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                    -1.0,
                    1.0,
                )
            )
            return np.degrees(angle_rad)

        nfa: float = calculate_angle(
            landmarks[lateral.LATERAL_LM_SOFT_TISSUE_NASION],
            landmarks[lateral.LATERAL_LM_SOFT_TISSUE_GLABELLA],
            landmarks[lateral.LATERAL_LM_NASAL_TIP],
        )

        nla: float = calculate_angle(
            landmarks[lateral.LATERAL_LM_SUBNASAL_POINT],
            landmarks[lateral.LATERAL_LM_NASAL_TIP],
            landmarks[lateral.LATERAL_LM_SOFT_TISSUE_POGONION],
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

        if render and render_nfa:
            face.arrow(
                landmarks[lateral.LATERAL_LM_SOFT_TISSUE_NASION],
                landmarks[lateral.LATERAL_LM_SOFT_TISSUE_GLABELLA],
                thickness=2,
                apt1=False,
            )
            face.arrow(
                landmarks[lateral.LATERAL_LM_SOFT_TISSUE_NASION],
                landmarks[lateral.LATERAL_LM_NASAL_TIP],
                thickness=2,
                apt1=False,
            )
            pt = (
                landmarks[lateral.LATERAL_LM_SOFT_TISSUE_NASION][0] + 10,
                landmarks[lateral.LATERAL_LM_SOFT_TISSUE_NASION][1],
            )
            txt_nfa: str = f"NFA={nfa:.2f}"
            face.write_text_sq(pt, txt_nfa, mark="o", up=10)

        if render and render_nla:
            face.arrow(
                landmarks[lateral.LATERAL_LM_SUBNASAL_POINT],
                landmarks[lateral.LATERAL_LM_NASAL_TIP],
                thickness=2,
                apt1=False,
            )
            face.arrow(
                landmarks[lateral.LATERAL_LM_SUBNASAL_POINT],
                landmarks[lateral.LATERAL_LM_SOFT_TISSUE_POGONION],
                thickness=2,
                apt1=False,
            )
            pt = (
                landmarks[lateral.LATERAL_LM_SUBNASAL_POINT][0] + 10,
                landmarks[lateral.LATERAL_LM_SUBNASAL_POINT][1],
            )
            txt_nla: str = f"NLA={nla:.2f}"
            face.write_text_sq(pt, txt_nla, mark="o", up=10)

        return filter_measurements(
            {"nn": nn, "nm": nm, "np": np_val, "nfa": nfa, "nla": nla}, self.items
        )
