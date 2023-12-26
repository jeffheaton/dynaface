import numpy as np


def filter(data, items):
    result = {}
    for item in items:
        result[item.name] = data[item.name]
    return result


def all_measures():
    return [
        AnalyzeFAI(),
        AnalyzeOralCommissureExcursion(),
        AnalyzeBrows(),
        AnalyzeDentalArea(),
        AnalyzeEyeArea(),
    ]


class MeasureItem:
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def __str__(self):
        return f"(name={self.name},enabled={self.enabled})"


class MeasureBase:
    def __init__(self):
        self.enabled = True
        self.items = []

    def set_item_enabled(self, name, enabled):
        for item in self.items:
            if item.name == name:
                item.enabled = enabled

    def is_enabled(self, name):
        for item in self.items:
            if item.name == name:
                return item.enabled
        return True


class AnalyzeFAI(MeasureBase):
    def __init__(self) -> None:
        self.enabled = True
        self.items = [MeasureItem("fai")]

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
            txt = f"FAI={fai:.2f} mm"
            pos = face.analyze_next_pt(txt)
            face.write_text(pos, txt)
        return filter({"fai": fai}, (self.items))


class AnalyzeOralCommissureExcursion(MeasureBase):
    def __init__(self) -> None:
        self.enabled = True
        self.items = [MeasureItem("oce.a"), MeasureItem("oce.b")]

    def abbrev(self):
        return "Oral CE"

    def calc(self, face, render=True):
        render2_a = self.is_enabled("oce.a")
        render2_b = self.is_enabled("oce.b")
        oce_a = face.measure(
            face.landmarks[76], face.landmarks[85], render=(render & render2_a), dir="l"
        )
        oce_b = face.measure(
            face.landmarks[82], face.landmarks[85], render=(render & render2_b), dir="r"
        )
        return filter({"oce.a": oce_a, "oce.b": oce_b}, self.items)


class AnalyzeBrows(MeasureBase):
    def __init__(self) -> None:
        self.enabled = True
        self.items = [MeasureItem("brow.d")]

    def abbrev(self):
        return "Brow"

    def calc(self, face, render=True):
        render2 = self.is_enabled("brow.d")

        # left brow
        contours = [
            face.landmarks[34],
            face.landmarks[35],
            face.landmarks[36],
            face.landmarks[37],
        ]

        contours = np.array(contours)
        x = contours[:, 0]
        y = contours[:, 1]
        left_brow_idx = np.argmin(y)
        left_brow_y = y[left_brow_idx]
        left_brow_x = x[left_brow_idx]
        if render & render2:
            face.arrow((left_brow_x, left_brow_y), (1024, left_brow_y), apt2=False)

        # right brow
        contours = [
            face.landmarks[42],
            face.landmarks[43],
            face.landmarks[44],
            face.landmarks[45],
        ]

        contours = np.array(contours)
        x = contours[:, 0]
        y = contours[:, 1]
        right_brow_idx = np.argmin(y)
        right_brow_y = y[right_brow_idx]
        right_brow_x = x[right_brow_idx]

        # Diff
        diff = abs(left_brow_y - right_brow_y) * face.pix2mm
        if render & render2:
            face.arrow((right_brow_x, right_brow_y), (1024, right_brow_y), apt2=False)
            txt = f"d.brow={diff:.2f} mm"
            m = face.calc_text_size(txt)

            face.write_text(
                (face.width - (m[0][0] + 5), min(left_brow_y, right_brow_y) - 10),
                txt,
            )

        return filter({"brow.d": diff}, self.items)


class AnalyzeDentalArea(MeasureBase):
    def __init__(self) -> None:
        self.enabled = True
        self.items = [MeasureItem("dental_area")]

    def abbrev(self):
        return "Dental Display"

    def calc(self, face, render=True):
        render2 = self.is_enabled("dental_area")

        contours = [
            face.landmarks[88],
            face.landmarks[89],
            face.landmarks[90],
            face.landmarks[91],
            face.landmarks[92],
            face.landmarks[93],
            face.landmarks[94],
            face.landmarks[95],
        ]

        contours = np.array(contours)  # contours = contours*face.pix2mm
        dental_area = face.measure_polygon(
            contours, face.pix2mm, render=(render & render2)
        )
        if render & render2:
            txt = f"dental={round(dental_area,2)}mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)
        return filter({"dental_area": dental_area}, self.items)


class AnalyzeEyeArea(MeasureBase):
    def __init__(self) -> None:
        self.enabled = True
        self.items = [
            MeasureItem("eye.l"),
            MeasureItem("eye.r"),
            MeasureItem("eye.d"),
            MeasureItem("eye.rlr"),
            MeasureItem("eye.rrl"),
        ]

    def abbrev(self):
        return "Eye Area"

    def calc(self, face, render=True):
        render2_eye_l = self.is_enabled("eye.l")
        render2_eye_r = self.is_enabled("eye.r")
        render2_eye_d = self.is_enabled("eye.d")
        render2_eye_rlr = self.is_enabled("eye.rlr")
        render2_eye_rrl = self.is_enabled("eye.rrl")

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
        if right_eye_area == 0:
            eye_ratio_lr = 0
        else:
            eye_ratio_lr = round(left_eye_area / right_eye_area, 2)

        if left_eye_area == 0:
            eye_ratio_rl = 0
        else:
            eye_ratio_rl = round(right_eye_area / left_eye_area, 2)

        if render & render2_eye_r:
            face.write_text_sq(
                (face.landmarks[66][0] - 150, face.landmarks[66][1] + 20),
                f"R={round(right_eye_area,2)}mm",
            )

        if render & render2_eye_l:
            face.write_text_sq(
                (face.landmarks[74][0] - 50, face.landmarks[74][1] + 20),
                f"L={round(left_eye_area,2)}mm",
            )

        if render & render2_eye_d:
            txt = f"d.eye={round(eye_area_diff,2)}mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render & render2_eye_rlr:
            txt = f"rlr.eye={round(eye_ratio_lr,2)}mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        if render & render2_eye_rrl:
            txt = f"rrl.eye={round(eye_ratio_rl,2)}mm"
            pos = face.analyze_next_pt(txt)
            face.write_text_sq(pos, txt)

        return filter(
            {
                "eye.l": left_eye_area,
                "eye.r": right_eye_area,
                "eye.d": eye_area_diff,
                "eye.rlr": eye_ratio_lr,
                "eye.rrl": eye_ratio_rl,
            },
            self.items,
        )
