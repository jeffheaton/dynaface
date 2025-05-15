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
    Tip Projection: (AT, NT, AT/NT)
    """

    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.items = [
            MeasureItem("nfa"),
            MeasureItem("nla"),
            MeasureItem("tip_proj"),
        ]
        self.is_frontal = False
        self.is_lateral = True
        self.sync_items()

    def abbrev(self) -> str:
        return "Lateral Measures"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        render_nfa: bool = self.is_enabled("nfa")
        render_nla: bool = self.is_enabled("nla")

        if not face.lateral:
            return {}

        landmarks: Any = face.lateral_landmarks

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

        # Tip projection measurements
        pt_A = landmarks[lateral.LATERAL_LM_NASAL_TIP]
        pt_N = landmarks[lateral.LATERAL_LM_SOFT_TISSUE_NASION]
        pt_T = (pt_N[0], pt_A[1])

        AT = util.euclidean_distance(pt_A, pt_T)
        NT = util.euclidean_distance(pt_N, pt_T)
        AT_NT_ratio = AT / NT if NT != 0 else float("nan")

        if render and self.is_enabled("tip_proj"):
            face.arrow(pt_A, pt_T, thickness=2, apt1=False)
            face.arrow(pt_N, pt_T, thickness=2, apt1=False)

            face.write_text(pt_A, "A")
            face.write_text(pt_N, "N")
            face.write_text(pt_T, "T")

            mid_txt = ((pt_A[0] + pt_T[0]) // 2 + 5, pt_A[1] + 25)
            face.write_text(mid_txt, f"AT={AT:.2f}")

            mid_txt_nt = ((pt_N[0] + pt_T[0]) // 2 + 5, (pt_N[1] + pt_T[1]) // 2)
            face.write_text(mid_txt_nt, f"NT={NT:.2f}")

            mid_ratio = (pt_T[0] + 10, pt_T[1] + 10)
            ratio_txt = f"AT/NT={AT_NT_ratio:.2f}"

            pos = face.analyze_next_pt(ratio_txt)
            face.write_text(pos, ratio_txt)

        return filter_measurements(
            {
                "nfa": nfa,
                "nla": nla,
                "at": AT,
                "nt": NT,
                "at_nt": AT_NT_ratio,
            },
            self.items,
        )
