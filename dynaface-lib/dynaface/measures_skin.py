import logging
from typing import Any, Dict

import numpy as np
from scipy.stats import mode
import math

from dynaface import facial, util
from dynaface.measures_base import MeasureBase, MeasureItem, filter_measurements
import colorsys

USE_MODE = False  # Set to False to use mean averaging instead

logger = logging.getLogger(__name__)

VON_LUSCHAN_RGB = {
    1: (255, 223, 196),
    2: (255, 224, 204),
    3: (255, 209, 170),
    4: (255, 196, 156),
    5: (255, 185, 122),
    6: (242, 166, 94),
    7: (234, 194, 156),
    8: (227, 178, 126),
    9: (223, 166, 122),
    10: (216, 148, 99),
    11: (208, 131, 81),
    12: (198, 120, 63),
    13: (187, 109, 50),
    14: (181, 101, 42),
    15: (167, 90, 37),
    16: (156, 80, 34),
    17: (143, 71, 30),
    18: (129, 64, 22),
    19: (118, 57, 22),
    20: (107, 50, 19),
    21: (96, 48, 18),
    22: (88, 43, 16),
    23: (77, 39, 14),
    24: (67, 34, 12),
    25: (58, 30, 10),
    26: (49, 26, 8),
    27: (41, 23, 6),
    28: (33, 19, 6),
    29: (26, 15, 5),
    30: (21, 12, 4),
    31: (18, 10, 3),
    32: (15, 8, 2),
    33: (12, 6, 2),
    34: (9, 5, 2),
    35: (6, 3, 2),
    36: (3, 2, 1),
}


class AnalyzeSkinTone(MeasureBase):
    """
    Analyze the skin toke using the Fitzpatrick and Von Luschan skin tone measurement.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.items = [MeasureItem("hsv")]
        self.is_frontal = True
        self.sync_items()

    def abbrev(self) -> str:
        return "Skin Tone"

    USE_MODE = True  # Set to False to use mean averaging instead

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        results = {}

        if render and self.is_enabled("hsv"):
            try:
                # Validate all required landmarks exist
                for idx in [2, 3, 30, 55, 59]:
                    if idx >= len(face.landmarks):
                        logger.warning(
                            f"Missing landmark index {idx}, skipping skin tone analysis."
                        )
                        return results

                square_area = face.width * face.height * 0.025
                square_size = int(math.sqrt(square_area))

                offset_x = int(face.width * 0.02)
                offset_y = int(face.height * 0.02)

                top_left = (face.width - offset_x - square_size, offset_y)
                bottom_right = (face.width - offset_x, offset_y + square_size)

                # Sample cheeks (two rectangles)
                top_left1 = (face.landmarks[3][0], face.landmarks[2][1])
                bottom_right1 = (face.landmarks[55][0], face.landmarks[55][1])

                top_left2 = (face.landmarks[30][0], face.landmarks[59][1])
                bottom_right2 = (face.landmarks[59][0], face.landmarks[30][1])

                sample1 = face.sample_rectangle(top_left1, bottom_right1)
                sample2 = face.sample_rectangle(top_left2, bottom_right2)

                if sample1.size == 0 or sample2.size == 0:
                    logger.warning(
                        "One or both sample regions are empty. Skipping skin tone rendering."
                    )
                    return results

                combined_samples = np.vstack([sample1, sample2])

                if combined_samples.size == 0:
                    logger.warning(
                        "Combined sample region is empty. Skipping skin tone rendering."
                    )
                    return results

                if USE_MODE:
                    color_rgb = tuple(
                        int(c)
                        for c in mode(combined_samples, axis=0, keepdims=False).mode
                    )
                else:
                    color_rgb = tuple(int(c) for c in np.mean(combined_samples, axis=0))

                # Draw the rectangle with the average skin color
                face.rectangle(top_left, bottom_right, color=color_rgb, filled=True)
                face.rectangle(top_left, bottom_right, color=(0, 0, 0), filled=False)

                # Convert RGB to HSV for display
                r_norm, g_norm, b_norm = [c / 255.0 for c in color_rgb]
                h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
                hue_deg = int(h * 360)
                sat_pct = int(s * 100)
                bri_pct = int(v * 100)

                lines = [
                    f"HUE: {hue_deg}",
                    f"SAT: {sat_pct}",
                    f"BRT: {bri_pct}",
                ]

                text_sizes = [face.calc_text_size(line)[0] for line in lines]
                total_text_height = sum(h for _, h in text_sizes) + 5 * (len(lines) - 1)

                current_y = top_left[1] + (square_size - total_text_height) // 2

                for line, (text_w, text_h) in zip(lines, text_sizes):
                    text_x = top_left[0] + (square_size - text_w) // 2
                    text_pos = (text_x, current_y + text_h)
                    face.write_text(text_pos, line, color=(255, 255, 255))
                    current_y += text_h + 5

            except Exception as e:
                logger.exception(f"Unexpected error during skin tone analysis: {e}")

        return filter_measurements(results, self.items)
