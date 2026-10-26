"""Measures of a lateral (side-profile) view."""

import logging
import math
from typing import Any, Dict
import numpy as np

from dynaface import lateral, util

logger = logging.getLogger(__name__)

from dynaface.measures_base import MeasureBase, MeasureItem, select_outputs


class AnalyzeLateral(MeasureBase):
    """
    Measures of a side profile, from the landmarks found along its outline.

    Outputs:
        - `nfa`: Nasofrontal angle, in degrees, at the soft tissue nasion between
          the glabella and the nasal tip.
        - `nla`: Nasolabial angle, in degrees, at the subnasal point between the
          nasal tip and the soft tissue pogonion.
        - `at`: Tip projection: the horizontal distance, in pixels, from the nasal
          tip (A) to point T, which is level with the tip and directly below the
          nasion.
        - `nt`: The vertical distance, in pixels, from the nasion (N) to T.
        - `at_nt`: The ratio AT/NT.

    The tip_proj item only switches the AT and NT drawing on and off; it has no
    value of its own.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.items = [
            MeasureItem("nfa"),
            MeasureItem("nla"),
            MeasureItem("tip_proj", display_only=True),  # draws AT/NT
            MeasureItem("at"),
            MeasureItem("nt"),
            MeasureItem("at_nt"),
        ]
        self.is_frontal = False
        self.is_lateral = True
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the lateral measures.

        Returns:
            str: Abbreviation string.
        """
        return "Lateral Measures"

    def calc(self, face: Any, render: bool = True) -> Dict[str, float]:
        """
        Calculate the lateral measures, drawing the enabled ones when render is
        True.

        Args:
            face (Any): Face object with its lateral_landmarks set.
            render (bool): Whether to render the measurements visually.

        Returns:
            Dict[str, float]: nfa, nla, at, nt and at_nt, with NaN where a landmark
                wasn't found, or an empty dict for a face that isn't lateral.
        """
        render_nfa: bool = self.is_enabled("nfa")
        render_nla: bool = self.is_enabled("nla")

        if not face.lateral:
            return {}

        landmarks: Any = face.lateral_landmarks

        def missing(pt: Any) -> bool:
            # The lateral finder marks a landmark it couldn't find as (-1, -1). Its x
            # is then shifted into image space like the others, but y stays -1.
            return bool(pt[1] < 0)

        def calculate_angle(pt_center: Any, pt1: Any, pt2: Any) -> float:
            # Undefined (NaN) when a point is missing or a side has zero length.
            if missing(pt_center) or missing(pt1) or missing(pt2):
                return float("nan")
            v1 = np.array(pt1, dtype=float) - np.array(pt_center, dtype=float)
            v2 = np.array(pt2, dtype=float) - np.array(pt_center, dtype=float)
            n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if n1 == 0 or n2 == 0:
                return float("nan")
            cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            return float(np.degrees(np.arccos(cos_angle)))

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

        if render and render_nfa and not math.isnan(nfa):
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

        if render and render_nla and not math.isnan(nla):
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

        if missing(pt_A) or missing(pt_N):
            AT = NT = AT_NT_ratio = float("nan")
        else:
            AT = util.euclidean_distance(pt_A, pt_T)
            NT = util.euclidean_distance(pt_N, pt_T)
            AT_NT_ratio = AT / NT if NT != 0 else float("nan")

        if render and self.is_enabled("tip_proj") and not math.isnan(AT):
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

        return select_outputs(
            {
                "nfa": nfa,
                "nla": nla,
                "at": AT,
                "nt": NT,
                "at_nt": AT_NT_ratio,
            },
            self.items,
        )
