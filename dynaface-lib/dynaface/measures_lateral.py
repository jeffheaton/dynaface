import logging
from typing import Any, Dict

from dynaface import lateral

logger = logging.getLogger(__name__)

from dynaface.measures_base import MeasureBase, MeasureItem, filter_measurements


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
