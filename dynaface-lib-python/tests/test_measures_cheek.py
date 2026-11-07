"""
Tests for dynaface.measures_cheek.AnalyzeCheekMeasures on mock faces: the six
items, the declared outputs, render independence, the drawing calls and the
text column on frontal, quarter and lateral faces, each item gating exactly its
own drawing; and the Landmarks measure's "Cheek Landmarks" item, which reports
and draws the cheek landmarks' coordinates. No models: the frontal face is img4's
pipeline landmarks from tests_data/cheek_landmarks.json, the quarter face the
yaw45 render's raw landmarks with its outline traced from the render's own
background color, and the contour fallback the app's three-quarter sketch.
"""

import dataclasses
import json
import math
import os
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface import cheek  # noqa: E402
from dynaface import facial  # noqa: E402
from dynaface import measures  # noqa: E402
from dynaface import video  # noqa: E402
from dynaface import cheek_outline as co  # noqa: E402
from dynaface import measures_cheek as mc  # noqa: E402
from dynaface.const import Pose  # noqa: E402
from dynaface.measures_cheek import AnalyzeCheekMeasures  # noqa: E402
from dynaface.measures_frontal import AnalyzeLandmarks  # noqa: E402

TESTS_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests_data"
)
LANDMARKS_JSON = os.path.join(TESTS_DATA, "cheek_landmarks.json")
RENDERS_JSON = os.path.join(TESTS_DATA, "cheek", "cheek_renders.json")
RENDERS_DIR = os.path.join(TESTS_DATA, "cheek")

# The six items, in order, with the views they suit and their outputs.
ITEMS = [
    (
        "Zygion width",
        (Pose.FRONTAL,),
        [
            "zy.r",
            "zy.l",
            "zy.w",
            "zy.ratio",
            "zy.ratio.ok",
            "zy.r.clip",
            "zy.l.clip",
            "zy.r.ok",
            "zy.l.ok",
            "zy.ok",
        ],
    ),
    (
        "Reference points",
        (Pose.FRONTAL,),
        ["cr.r.d", "cr.l.d", "cr.ok", "sr.ok"],
    ),
    (
        "Far cheek outline",
        (Pose.QUARTER,),
        ["cheek.far", "cheek.mask", "cheek.nose", "cheek.nose.gap", "cheek.scale"],
    ),
    (
        "Malar apex",
        (Pose.QUARTER,),
        [
            "apex.r.h",
            "apex.l.h",
            "apex.r.dev",
            "apex.l.dev",
            "apex.r.prior",
            "apex.l.prior",
            "apex.r.near_nose",
            "apex.l.near_nose",
            "apex.ok",
        ],
    ),
    (
        "Submalar point",
        (Pose.QUARTER,),
        [
            "sub.r.h",
            "sub.l.h",
            "sub.r.depth",
            "sub.l.depth",
            "sub.r.depth.px",
            "sub.l.depth.px",
            "sub.r.present",
            "sub.l.present",
            "sub.ok",
        ],
    ),
    (
        "Head turn",
        (Pose.FRONTAL, Pose.QUARTER),
        ["cheek.yaw", "cheek.yaw.model", "cheek.pitch", "cheek.pitch.ok"],
    ),
]
ITEM_NAMES = [name for name, _, _ in ITEMS]
OUTPUTS_OF = {name: outputs for name, _, outputs in ITEMS}
ZYGION, REFERENCE, OUTLINE, APEX, SUBMALAR, HEAD_TURN = ITEM_NAMES
# The 41 outputs, in CSV order: the items' outputs, item by item.
EXPECTED_NAMES = [name for _, _, outputs in ITEMS for name in outputs]
# The frontal items' outputs (NaN on a quarter face), the quarter items' (NaN on
# a frontal face) and the head turn's (numeric in either view; cheek.yaw is NaN
# without a pupil frame).
FRONTAL_NAMES = OUTPUTS_OF[ZYGION] + OUTPUTS_OF[REFERENCE]
OBLIQUE_NAMES = OUTPUTS_OF[OUTLINE] + OUTPUTS_OF[APEX] + OUTPUTS_OF[SUBMALAR]
SHARED_NAMES = OUTPUTS_OF[HEAD_TURN]
OK_FLAGS = [name for name in EXPECTED_NAMES if name.endswith(".ok")]
# The cheek landmarks' coordinates: the Landmarks measure's Cheek Landmarks item.
COORD_NAMES = list(cheek.CHEEK_LANDMARK_OUTPUTS)


def coords(face):
    """The cheek landmarks' 20 coordinates the Landmarks measure reports for a
    face (cheek.locate() on a mock face's landmarks, view and outline)."""
    result = AnalyzeLandmarks().calc(face, render=False)
    return {name: result[name] for name in COORD_NAMES}


def only(*names):
    """A Cheek Measures measure with only these items enabled."""
    measure = AnalyzeCheekMeasures()
    for item in measure.items:
        item.enabled = item.name in names
    return measure


def _same(a, b):
    """Equal, with NaN equal to NaN."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def _runtime_warnings(caught):
    """The RuntimeWarnings among the warnings recorded, as strings."""
    return [str(w.message) for w in caught if w.category is RuntimeWarning]


def flagged(*reasons):
    """The text column lines that say why a three-quarter result is flagged."""
    return [mc.FLAG_PREFIX + reason for reason in reasons]


class MockFace:
    """
    A face stub for AnalyzeCheekMeasures without loaded models: the landmarks,
    the view, the head pose, the scale, the image width (the labels keep inside
    it), the outline arrays of a quarter face, and every drawing call recorded.
    """

    def __init__(
        self,
        landmarks,
        lateral=False,
        quarter=False,
        far_left=False,
        yaw=0.0,
        pitch=0.0,
        pix2mm=1.0,
        outline=(None, None, None),
        width=1024,
    ):
        self.landmarks = landmarks
        self.width = width
        self.lateral = lateral
        self.quarter = quarter
        self.far_left = far_left
        self.yaw = yaw
        self.pitch = pitch
        self.roll = 0.0
        self.pix2mm = pix2mm
        self.cheek_outline_x, self.cheek_outline_y, self.cheek_outline_src = outline
        self.calls = []  # every drawing call, as (kind, ...)
        self.points = []  # every point handed to a drawing call
        self.column = []  # the text column's lines, in order
        self.analyze_y = 30

    def line(self, pt1, pt2, color=(255, 0, 0), thickness=3):
        self.points += [pt1, pt2]
        self.calls.append(("line", pt1, pt2, color, thickness))

    def circle(self, pt, color=(0, 0, 255), radius=None, thickness=-1):
        self.points.append(pt)
        self.calls.append(("circle", pt, color, radius, thickness))

    def polyline(self, points, color=(255, 0, 0), thickness=3):
        points = list(points)
        self.points += points
        self.calls.append(("polyline", points, color, thickness))

    def write_text(self, pos, txt, color=None, size=1, thick=1):
        self.points.append(pos)
        self.calls.append(("text", pos, txt, color, size, thick))

    def calc_text_size(self, txt, size=1, thick=1):
        return ((12 * len(txt), 20), 5)

    def analyze_next_pt(self, txt):
        self.column.append(txt)
        pos = (10, self.analyze_y)
        self.analyze_y += 40
        return pos

    def kinds(self, kind):
        """The recorded calls of one kind."""
        return [call for call in self.calls if call[0] == kind]

    def labels(self):
        """The texts written with a color: the point labels, not the column."""
        return [call[2] for call in self.kinds("text") if call[3] is not None]


class Fixtures:
    """The fixture faces, loaded once."""

    _images = None
    _renders = None

    @classmethod
    def images(cls):
        if cls._images is None:
            with open(LANDMARKS_JSON, encoding="utf-8") as f:
                cls._images = json.load(f)["images"]
        return cls._images

    @classmethod
    def renders(cls):
        if cls._renders is None:
            with open(RENDERS_JSON, encoding="utf-8") as f:
                cls._renders = json.load(f)
        return cls._renders

    @classmethod
    def frontal(cls, name="img4", **overrides):
        """A frontal face from an entry's pipeline landmarks and head pose."""
        pipeline = cls.images()[name]["pipeline"]
        landmarks = [tuple(int(v) for v in xy) for xy in pipeline["landmarks"]]
        kwargs = dict(
            yaw=pipeline["yaw"],
            pitch=pipeline["pitch"],
            pix2mm=pipeline["pix2mm"],
            width=pipeline["size"][0],
        )
        kwargs.update(overrides)
        return MockFace(landmarks, **kwargs)

    @classmethod
    def traced(cls, name):
        """
        A render's raw landmarks, far side and outline, traced from the mask
        the render's background color defines.
        """
        render = cls.renders()["renders"][name]
        background = np.array(cls.renders()["background"], dtype=np.uint8)
        path = os.path.join(RENDERS_DIR, render["file"])
        image = np.asarray(Image.open(path).convert("RGB"))
        foreground = np.any(image != background, axis=2)
        alpha = np.where(foreground, 255, 0).astype(np.uint8)
        raw = cls.images()[name]["raw"]
        landmarks = [tuple(int(v) for v in xy) for xy in raw["landmarks"]]
        far_left = render["far_side"] == "left"
        outline = co.trace_far_cheek(
            foreground, alpha, np.array(landmarks, dtype=np.float64), far_left
        )
        return landmarks, far_left, raw, outline

    @classmethod
    def quarter(cls, name="yaw45", **overrides):
        """A quarter face from a render: its raw landmarks and traced outline."""
        landmarks, far_left, raw, outline = cls.traced(name)
        kwargs = dict(
            quarter=True,
            far_left=far_left,
            yaw=raw["yaw"],
            pitch=raw["pitch"],
            outline=outline,
            width=cls.images()[name]["size"][0],
        )
        kwargs.update(overrides)
        return MockFace(landmarks, **kwargs)

    @classmethod
    def sketch(cls, **overrides):
        """
        The app's three-quarter sketch as a quarter face without an outline: the
        contour fallback. Its nose points to the image left, so the far cheek is
        the subject's right.
        """
        raw = cls.images()["quarter_sketch"]["raw"]
        landmarks = [tuple(int(v) for v in xy) for xy in raw["landmarks"]]
        kwargs = dict(
            quarter=True,
            far_left=False,
            yaw=raw["yaw"],
            pitch=raw["pitch"],
            width=cls.images()["quarter_sketch"]["size"][0],
        )
        kwargs.update(overrides)
        return MockFace(landmarks, **kwargs)

    @classmethod
    def lateral(cls):
        """A profile."""
        return cls.frontal(lateral=True)


class TestStructure(unittest.TestCase):
    def setUp(self):
        self.measure = AnalyzeCheekMeasures()

    def test_abbrev(self):
        self.assertEqual(self.measure.abbrev(), "Cheek Measures")

    def test_six_items_in_order(self):
        self.assertEqual(
            ITEM_NAMES,
            [
                "Zygion width",
                "Reference points",
                "Far cheek outline",
                "Malar apex",
                "Submalar point",
                "Head turn",
            ],
        )
        self.assertEqual([i.name for i in self.measure.items], ITEM_NAMES)
        self.assertEqual(list(mc.ITEM_NAMES), ITEM_NAMES)
        for item, (name, views, outputs) in zip(self.measure.items, ITEMS):
            self.assertFalse(item.display_only, name)
            self.assertEqual(item.output_names(), outputs, name)
            self.assertEqual(item.views, views, name)
        self.assertEqual(
            [(n, tuple(v), list(o)) for n, v, o in mc.ITEMS],
            [(n, v, o) for n, v, o in ITEMS],
        )

    def test_names_in_order(self):
        self.assertEqual(self.measure.outputs(), EXPECTED_NAMES)
        self.assertEqual(len(EXPECTED_NAMES), 41)
        self.assertEqual([len(outputs) for _, _, outputs in ITEMS], [10, 4, 5, 9, 9, 4])
        self.assertEqual(list(mc.OUTPUTS), EXPECTED_NAMES)
        # No coordinates (the Landmarks measure's) and no drawing switches.
        for name in COORD_NAMES:
            self.assertNotIn(name, EXPECTED_NAMES)
        for name in ITEM_NAMES:
            self.assertNotIn(name, EXPECTED_NAMES)

    def test_view_flags(self):
        self.assertTrue(self.measure.enabled)
        self.assertTrue(self.measure.is_frontal)
        self.assertTrue(self.measure.is_quarter)
        self.assertFalse(self.measure.is_lateral)
        self.assertTrue(self.measure.suits(lateral=False))
        self.assertTrue(self.measure.suits(lateral=False, quarter=True))
        self.assertFalse(self.measure.suits(lateral=True))
        for item, (name, views, _) in zip(self.measure.items, ITEMS):
            self.assertEqual(item.is_frontal, Pose.FRONTAL in views, name)
            self.assertEqual(item.is_quarter, Pose.QUARTER in views, name)
            self.assertFalse(item.is_lateral, name)
            self.assertTrue(item.enabled, name)
        # update_for_type() leaves each item on only where it suits the face.
        self.measure.update_for_type(lateral=False, quarter=True)
        self.assertEqual(
            [i.name for i in self.measure.items if i.enabled],
            [OUTLINE, APEX, SUBMALAR, HEAD_TURN],
        )
        self.measure.update_for_type(lateral=False)
        self.assertEqual(
            [i.name for i in self.measure.items if i.enabled],
            [ZYGION, REFERENCE, HEAD_TURN],
        )
        self.assertEqual(
            measures.get_all_items([self.measure]),
            FRONTAL_NAMES + SHARED_NAMES,
        )

    def test_registered_last(self):
        self.assertIsInstance(measures.all_measures()[-1], AnalyzeCheekMeasures)

    def test_pt_rounds_to_ints(self):
        self.assertEqual(mc._pt(np.array([1.5, 2.49])), (2, 2))
        self.assertEqual(mc._pt((-0.6, 10.0)), (-1, 10))
        for v in mc._pt(np.array([3.7, 4.2], dtype=np.float32)):
            self.assertIs(type(v), int)

    def test_outline_colors_cover_every_source(self):
        self.assertEqual(
            set(mc.OUTLINE_COLORS),
            {
                co.OUTLINE_SRC_CONTOUR,
                co.OUTLINE_SRC_MASK,
                co.OUTLINE_SRC_NOSE,
                co.OUTLINE_SRC_EDGE,
            },
        )
        self.assertEqual(len(set(mc.OUTLINE_COLORS.values())), 4)


class TestDeclaredOutputs(unittest.TestCase):
    """
    Every declared output comes back on a frontal and a quarter face, nothing on a
    profile, whether or not the items are enabled and whether or not it renders,
    with the same values either way.
    """

    def check(self, make_face, expected):
        for enabled in (True, False):
            measure = AnalyzeCheekMeasures()
            for item in measure.items:
                item.enabled = enabled
            results = {}
            for render in (False, True):
                face = make_face()
                with self.subTest(enabled=enabled, render=render):
                    result = measure.calc(face, render=render)
                    self.assertEqual(set(result), expected)
                    results[render] = result
                    if not render or not enabled:
                        self.assertEqual(face.calls, [])
            for name in expected:
                with self.subTest(enabled=enabled, name=name):
                    self.assertTrue(
                        _same(results[False][name], results[True][name]),
                        f"{name}: {results[False][name]} != {results[True][name]}",
                    )

    def test_frontal(self):
        self.check(Fixtures.frontal, set(EXPECTED_NAMES))

    def test_quarter(self):
        self.check(Fixtures.quarter, set(EXPECTED_NAMES))

    def test_quarter_fallback(self):
        self.check(Fixtures.sketch, set(EXPECTED_NAMES))

    def test_lateral(self):
        self.check(Fixtures.lateral, set())

    def test_every_value_is_a_float(self):
        for make_face in (Fixtures.frontal, Fixtures.quarter, Fixtures.sketch):
            result = AnalyzeCheekMeasures().calc(make_face(), render=False)
            for name, value in result.items():
                with self.subTest(face=make_face.__name__, name=name):
                    self.assertIs(type(value), float)
                    number = video._number(value)
                    self.assertIs(type(number), float)
                    self.assertTrue(_same(number, value))

    def test_flags_are_0_1_or_nan(self):
        # Only the two clip codes may read 2 (CLIP_BOTTOM); every other flag is
        # 0.0 or 1.0, or NaN where the view doesn't define it.
        for make_face in (Fixtures.frontal, Fixtures.quarter, Fixtures.sketch):
            result = AnalyzeCheekMeasures().calc(make_face(), render=False)
            for name in OK_FLAGS + [
                "zy.r.clip",
                "zy.l.clip",
                "apex.r.prior",
                "apex.l.prior",
                "apex.r.near_nose",
                "apex.l.near_nose",
                "sub.r.present",
                "sub.l.present",
                "cheek.far",
                "cheek.nose",
            ]:
                value = result[name]
                allowed = (0.0, 1.0, 2.0) if name.endswith(".clip") else (0.0, 1.0)
                with self.subTest(face=make_face.__name__, name=name):
                    self.assertTrue(
                        math.isnan(value) or value in allowed, (name, value)
                    )

    def test_too_few_landmarks_give_nan_not_an_exception(self):
        for make_face in (Fixtures.frontal, Fixtures.quarter):
            face = make_face()
            face.landmarks = face.landmarks[:50]
            result = AnalyzeCheekMeasures().calc(face, render=True)
            self.assertEqual(set(result), set(EXPECTED_NAMES))
            for name in FRONTAL_NAMES + OBLIQUE_NAMES:
                if name == "cheek.far":
                    continue
                self.assertTrue(math.isnan(result[name]), name)
            self.assertTrue(math.isnan(result["cheek.yaw"]))
            self.assertFalse(math.isnan(result["cheek.yaw.model"]))
            self.assertEqual(result["cheek.pitch.ok"], 1.0)
            # Nothing to draw but the pose line, and on a quarter face why.
            self.assertEqual(face.column[0][:9], "cheek yaw")
            self.assertIn("(flagged)", face.column[0])
            reasons = flagged("the far cheek could not be measured")
            self.assertEqual(face.column[1:], reasons if face.quarter else [])

    def test_non_finite_landmark_gives_nan_not_an_exception(self):
        """A NaN chin: no pupil frame, so both modules and cheek.yaw are NaN in
        either view, whether or not it renders, with no numpy warning, and the
        cheek landmarks agree (all NaN)."""
        for make_face, pose in (
            (Fixtures.frontal, Pose.FRONTAL),
            (Fixtures.quarter, Pose.QUARTER),
        ):
            for render in (False, True):
                face = make_face()
                face.landmarks = list(face.landmarks)
                x, _ = face.landmarks[cheek.LM_CHIN]
                face.landmarks[cheek.LM_CHIN] = (x, math.nan)
                with self.subTest(face=make_face.__name__, render=render):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result = AnalyzeCheekMeasures().calc(face, render=render)
                    self.assertEqual(_runtime_warnings(caught), [])
                    self.assertEqual(set(result), set(EXPECTED_NAMES))
                    for name in FRONTAL_NAMES + OBLIQUE_NAMES + ["cheek.yaw"]:
                        if name != "cheek.far":
                            self.assertTrue(math.isnan(result[name]), name)
                    self.assertTrue(
                        _same(result["cheek.far"], 1.0 if face.quarter else math.nan)
                    )
                    self.assertFalse(math.isnan(result["cheek.yaw.model"]))
                    self.assertEqual(result["cheek.pitch.ok"], 1.0)
                    if render:
                        # The column alone: the pose line, on a quarter face why.
                        reasons = flagged("the far cheek could not be measured")
                        self.assertEqual({call[0] for call in face.calls}, {"text"})
                        self.assertIn("(flagged)", face.column[0])
                        self.assertEqual(
                            face.column[1:], reasons if face.quarter else []
                        )
                    else:
                        self.assertEqual(face.calls, [])
                    located = cheek.locate_all(
                        face.landmarks,
                        pose,
                        face.far_left,
                        face.cheek_outline_x,
                        face.cheek_outline_y,
                        face.cheek_outline_src,
                    )
                    self.assertTrue(np.isnan(located).all())


class TestFrontal(unittest.TestCase):
    """The frontal core on img4's pipeline landmarks."""

    def setUp(self):
        self.measure = AnalyzeCheekMeasures()
        self.face = Fixtures.frontal()
        self.result = self.measure.calc(self.face, render=True)

    def test_zygion_width(self):
        self.assertAlmostEqual(self.result["zy.w"], 130.6, delta=0.2)
        self.assertAlmostEqual(self.result["zy.r"] + self.result["zy.l"], 130.6, 0)
        self.assertAlmostEqual(self.result["zy.r"], 65.5, delta=0.1)
        self.assertAlmostEqual(self.result["zy.l"], 65.1, delta=0.1)
        self.assertAlmostEqual(self.result["zy.ratio"], 65.06 / 65.53, delta=0.002)
        self.assertAlmostEqual(self.result["cr.r.d"], 28.8, delta=0.1)
        self.assertAlmostEqual(self.result["cr.l.d"], 28.8, delta=0.1)

    def test_values_follow_the_constructions(self):
        fc = cheek.frontal_constructions(self.face.landmarks)
        r = self.result
        c = coords(self.face)  # the Landmarks measure's Cheek Landmarks item
        self.assertEqual((c["zy.r.x"], c["zy.r.y"]), tuple(fc.zygion_r))
        self.assertEqual((c["zy.l.x"], c["zy.l.y"]), tuple(fc.zygion_l))
        self.assertEqual((c["cr.r.x"], c["cr.r.y"]), tuple(fc.cheek_ref_r))
        self.assertEqual((c["cr.l.x"], c["cr.l.y"]), tuple(fc.cheek_ref_l))
        self.assertEqual((c["sr.r.x"], c["sr.r.y"]), tuple(fc.submalar_ref_r))
        self.assertEqual((c["sr.l.x"], c["sr.l.y"]), tuple(fc.submalar_ref_l))
        self.assertEqual(r["zy.r"], fc.zygion_r_px * self.face.pix2mm)
        self.assertEqual(r["cr.l.d"], fc.cheek_ref_l_px * self.face.pix2mm)

    def test_gates_pass_on_img4(self):
        r = self.result
        for name in ("zy.r.ok", "zy.l.ok", "zy.ok", "zy.ratio.ok", "cr.ok", "sr.ok"):
            self.assertEqual(r[name], 1.0, name)
        self.assertEqual(r["zy.r.clip"], 0.0)
        self.assertEqual(r["zy.l.clip"], 0.0)
        self.assertEqual(r["cheek.pitch.ok"], 1.0)
        self.assertEqual(r["cheek.pitch"], 0.05)
        self.assertAlmostEqual(r["cheek.yaw"], -0.885, delta=0.01)
        self.assertEqual(r["cheek.yaw.model"], cheek.calibrate_yaw(1.08))
        # A raw 1.08: below the level head's reading (1.5) by less than the
        # table's first step, so the model reads a level head.
        self.assertAlmostEqual(r["cheek.yaw.model"], 0.0, delta=0.01)

    def test_oblique_outputs_nan(self):
        for name in OBLIQUE_NAMES:
            self.assertTrue(math.isnan(self.result[name]), name)

    def test_drawing_calls(self):
        face = self.face
        for point in face.points:
            self.assertIsInstance(point, tuple)
            self.assertEqual(len(point), 2)
            for v in point:
                self.assertIs(type(v), int, point)
        lines = face.kinds("line")
        # The level line, then the two canthus verticals.
        self.assertEqual(lines[0], ("line", (201, 558), (818, 558), mc.ORANGE, 2))
        self.assertEqual(len(lines), 3)
        for call in lines[1:]:
            self.assertEqual((call[3], call[4]), (mc.ORANGE, 3))
            self.assertEqual(call[1][0], call[2][0])  # vertical
            self.assertLess(call[1][1], call[2][1])  # canthus above the foot
        circles = face.kinds("circle")
        self.assertEqual(
            circles[:4],
            [
                ("circle", (240, 558), mc.WHITE, 9, -1),
                ("circle", (240, 558), mc.BLUE, 7, -1),
                ("circle", (778, 558), mc.WHITE, 9, -1),
                ("circle", (778, 558), mc.BLUE, 7, -1),
            ],
        )
        rings = circles[4:]
        self.assertEqual(len(rings), 4)
        for call in rings:
            self.assertEqual(call[2:], (mc.ORANGE, 7, 3))
        self.assertEqual(
            face.labels(),
            [
                "zygion",
                "zygion",
                "cheek ref",
                "submalar ref",
                "cheek ref",
                "submalar ref",
            ],
        )
        colors = [call[3] for call in face.kinds("text") if call[3] is not None]
        self.assertEqual(colors, [mc.BLUE] * 2 + [mc.ORANGE] * 4)
        # The right side's labels sit left of the point, the left side's right of
        # it: the zygion's above it, the reference points' beside their rings,
        # centered on the ring's row (the mock's text is 20 px tall).
        texts = [call for call in face.kinds("text") if call[3] is not None]
        self.assertLess(texts[0][1][0], 240)
        self.assertEqual(texts[1][1], (778 + 14, 558 - 22))
        self.assertEqual(texts[0][1], (240 - 12 * len("zygion") - 14, 558 - 22))
        for ring, text, outside_left in zip(
            rings, texts[2:], (True, True, False, False)
        ):
            (x, y), width = ring[1], 12 * len(text[2])
            expected_x = x - 14 - width if outside_left else x + 14
            self.assertEqual(text[1], (expected_x, y + 10), text[2])

    def test_text_column(self):
        self.assertEqual(
            self.face.column,
            [
                "zy.w=130.6 mm (R 65.5 / L 65.1, ratio 0.99)",
                "cheek ref depth R 28.8 / L 28.8 mm",
                "cheek yaw -0.9 (model 0.0) pitch 0.1",
            ],
        )
        # Three lines at most for any one face (the zygion, reference and pose
        # lines here; a quarter face's are the apex, submalar and pose lines).
        self.assertLessEqual(len(self.face.column), 3)
        column = [call for call in self.face.kinds("text") if call[3] is None]
        self.assertEqual([call[2] for call in column], self.face.column)
        self.assertEqual([call[1] for call in column], [(10, 30), (10, 70), (10, 110)])

    def test_items_off_draw_nothing(self):
        measure = only()
        face = Fixtures.frontal()
        result = measure.calc(face, render=True)
        self.assertEqual(face.calls, [])
        self.assertEqual(face.column, [])
        for name in EXPECTED_NAMES:
            self.assertTrue(_same(result[name], self.result[name]), name)

    def test_zygion_item_with_the_head_turn(self):
        measure = only(ZYGION, HEAD_TURN)
        face = Fixtures.frontal()
        measure.calc(face, render=True)
        self.assertEqual(len(face.kinds("line")), 1)
        self.assertEqual(len(face.kinds("circle")), 4)
        self.assertEqual(face.labels(), ["zygion", "zygion"])
        self.assertEqual(
            face.column,
            [
                "zy.w=130.6 mm (R 65.5 / L 65.1, ratio 0.99)",
                "cheek yaw -0.9 (model 0.0) pitch 0.1",
            ],
        )

    def test_reference_item_with_the_head_turn(self):
        measure = only(REFERENCE, HEAD_TURN)
        face = Fixtures.frontal()
        measure.calc(face, render=True)
        self.assertEqual(len(face.kinds("line")), 2)
        self.assertEqual(len(face.kinds("circle")), 4)
        self.assertEqual(len(face.labels()), 4)
        self.assertEqual(
            face.column,
            [
                "cheek ref depth R 28.8 / L 28.8 mm",
                "cheek yaw -0.9 (model 0.0) pitch 0.1",
            ],
        )

    def test_each_item_gates_exactly_its_own_drawing(self):
        """Each item alone draws its own share of the full drawing, in the same
        calls, and writes its own column line; the values never change."""
        full = self.face.calls
        drawn_lines = [c for c in full if c[0] == "line"]
        lines = drawn_lines[:1], drawn_lines[1:]  # the level line; the verticals
        circles = [c for c in full if c[0] == "circle"]
        texts = [c for c in full if c[0] == "text" and c[3] is not None]
        expected = {
            ZYGION: (lines[0] + circles[:4] + texts[:2], [self.face.column[0]]),
            REFERENCE: (lines[1] + circles[4:] + texts[2:], [self.face.column[1]]),
            OUTLINE: ([], []),
            APEX: ([], []),
            SUBMALAR: ([], []),
            HEAD_TURN: ([], [self.face.column[2]]),
        }
        for name in ITEM_NAMES:
            with self.subTest(item=name):
                face = Fixtures.frontal()
                result = only(name).calc(face, render=True)
                drawn = [c for c in face.calls if c[0] != "text" or c[3] is not None]
                calls, column = expected[name]
                self.assertEqual(sorted(map(repr, drawn)), sorted(map(repr, calls)))
                self.assertEqual(face.column, column)
                self.assertEqual(list(result), EXPECTED_NAMES)
                for output in EXPECTED_NAMES:
                    self.assertTrue(_same(result[output], self.result[output]), output)

    def test_head_turn_line_alone(self):
        # The pose line is the head turn's, whatever else is on or off.
        face = Fixtures.frontal()
        only(HEAD_TURN).calc(face, render=True)
        self.assertEqual(face.column, ["cheek yaw -0.9 (model 0.0) pitch 0.1"])
        self.assertEqual(len(face.calls), 1)
        face = Fixtures.frontal()
        only(ZYGION, REFERENCE).calc(face, render=True)
        self.assertEqual(face.column, self.face.column[:2])

    def test_render_false_draws_nothing(self):
        face = Fixtures.frontal()
        result = AnalyzeCheekMeasures().calc(face, render=False)
        self.assertEqual(face.calls, [])
        for name in EXPECTED_NAMES:
            self.assertTrue(_same(result[name], self.result[name]), name)

    def test_pitch_25_fails_only_pitch_ok(self):
        face = Fixtures.frontal(pitch=25.0)
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(result["cheek.pitch"], 25.0)
        self.assertEqual(result["cheek.pitch.ok"], 0.0)
        for name in EXPECTED_NAMES:
            if name in ("cheek.pitch", "cheek.pitch.ok"):
                continue
            self.assertTrue(_same(result[name], self.result[name]), name)
        self.assertEqual(
            face.column[-1], "cheek yaw -0.9 (model 0.0) pitch 25.0 (flagged)"
        )
        self.assertEqual(face.column[0], self.face.column[0])  # zy.ok still 1

    def test_nan_yaw_fails_every_yaw_gate(self):
        face = Fixtures.frontal(yaw=math.nan)
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertTrue(math.isnan(result["cheek.yaw.model"]))
        self.assertFalse(math.isnan(result["cheek.yaw"]))  # from the landmarks
        for name in ("zy.r.ok", "zy.l.ok", "zy.ok", "zy.ratio.ok", "cr.ok", "sr.ok"):
            self.assertEqual(result[name], 0.0, name)
        self.assertEqual(result["cheek.pitch.ok"], 1.0)  # pitch is its own flag
        self.assertAlmostEqual(result["zy.w"], self.result["zy.w"])  # still reported
        self.assertIn("(flagged)", face.column[0])
        self.assertIn("(flagged)", face.column[-1])
        # Gray dots and rings for a failed gate.
        self.assertEqual(
            [c[2] for c in face.kinds("circle")],
            [mc.WHITE, mc.GRAY] * 2 + [mc.GRAY] * 4,
        )
        # With the pitch unknown too, every flag is 0.
        face = Fixtures.frontal(yaw=math.nan, pitch=math.nan)
        result = AnalyzeCheekMeasures().calc(face, render=False)
        flags = [result[name] for name in OK_FLAGS if not math.isnan(result[name])]
        self.assertEqual(len(flags), 7)
        self.assertEqual(set(flags), {0.0})

    def test_turned_head_is_reported_and_flagged(self):
        # The 10 degree render: the frontal core is computed, with the gates 0.
        pipeline = Fixtures.images()["yaw10"]["pipeline"]
        self.assertEqual(pipeline["pose"], "frontal")
        face = Fixtures.frontal("yaw10")
        result = AnalyzeCheekMeasures().calc(face, render=False)
        self.assertAlmostEqual(result["cheek.yaw"], 10.0, delta=2.0)
        self.assertFalse(math.isnan(result["zy.w"]))
        for name in ("zy.r.ok", "zy.l.ok", "zy.ok", "zy.ratio.ok", "cr.ok", "sr.ok"):
            self.assertEqual(result[name], 0.0, name)

    def test_disagreeing_model_yaw_fails_the_gate(self):
        face = Fixtures.frontal(yaw=28.0)  # the model says 45 degrees; the landmarks 0
        result = AnalyzeCheekMeasures().calc(face, render=False)
        self.assertAlmostEqual(result["cheek.yaw.model"], 44.5, delta=0.5)
        for name in ("zy.r.ok", "zy.l.ok", "zy.ok", "zy.ratio.ok", "cr.ok", "sr.ok"):
            self.assertEqual(result[name], 0.0, name)

    def test_clipped_zygion(self):
        # The right contour's top (0-4) pushed below the level: the level passes
        # above it, the point is landmark 0 and only that side is flagged.
        face = Fixtures.frontal()
        face.landmarks = [
            (x, y + 250) if i < 5 else (x, y) for i, (x, y) in enumerate(face.landmarks)
        ]
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(result["zy.r.clip"], float(cheek.CLIP_TOP))
        self.assertEqual(result["zy.l.clip"], 0.0)
        c = coords(face)
        self.assertEqual((c["zy.r.x"], c["zy.r.y"]), face.landmarks[0])
        self.assertEqual(result["zy.r.ok"], 0.0)
        self.assertEqual(result["zy.l.ok"], 1.0)
        self.assertEqual(result["zy.ok"], 0.0)
        self.assertEqual(result["zy.ratio.ok"], 0.0)
        self.assertEqual(result["cr.ok"], 1.0)
        self.assertEqual(face.labels()[:2], ["zygion (clipped)", "zygion"])
        self.assertEqual(
            [c[2] for c in face.kinds("circle")[:4]],
            [mc.WHITE, mc.GRAY, mc.WHITE, mc.BLUE],
        )
        self.assertTrue(face.column[0].endswith(" (flagged)"))
        self.assertTrue(face.column[-1].endswith(" (flagged)"))

    def test_frontal_cheek_accessor_is_read(self):
        # A face with frontal_cheek() hands over its own constructions: here the
        # ones of the landmarks shifted 100 px right, a sentinel the landmarks on
        # the face would never give.
        face = Fixtures.frontal()
        shifted = [(x + 100, y) for x, y in face.landmarks]
        sentinel = cheek.frontal_constructions(shifted)
        face.frontal_cheek = lambda: sentinel
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(result["zy.w"], self.result["zy.w"])
        self.assertEqual(face.kinds("circle")[0][1], (240 + 100, 558))
        # And one whose accessor has nothing gives NaN, whatever the landmarks.
        face = Fixtures.frontal()
        face.frontal_cheek = lambda: None
        result = AnalyzeCheekMeasures().calc(face, render=True)
        for name in FRONTAL_NAMES:
            self.assertTrue(math.isnan(result[name]), name)
        self.assertFalse(math.isnan(result["cheek.yaw"]))
        self.assertEqual(
            face.column, ["cheek yaw -0.9 (model 0.0) pitch 0.1 (flagged)"]
        )

    def test_coincident_pupils_give_nan(self):
        face = Fixtures.frontal()
        face.landmarks = list(face.landmarks)
        face.landmarks[97] = face.landmarks[96]
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(set(result), set(EXPECTED_NAMES))
        for name in FRONTAL_NAMES + OBLIQUE_NAMES + ["cheek.yaw"]:
            self.assertTrue(math.isnan(result[name]), name)
        self.assertEqual(result["cheek.pitch.ok"], 1.0)

    def test_nan_canthus_never_raises_when_rendering(self):
        """A NaN lateral canthus: calc(render=True) returns wherever
        calc(render=False) does, with the same values, and nothing raises."""
        # In the landmarks: no frame at all, so nothing frontal is reported or
        # drawn, and cheek.yaw is NaN too.
        results, faces = {}, {}
        for render in (False, True):
            face = Fixtures.frontal()
            face.landmarks = list(face.landmarks)
            face.landmarks[cheek.LM_CANTHUS_L] = (math.nan, math.nan)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results[render] = AnalyzeCheekMeasures().calc(face, render=render)
            self.assertEqual(_runtime_warnings(caught), [])
            faces[render] = face
        self.assertEqual(set(results[True]), set(EXPECTED_NAMES))
        for name in EXPECTED_NAMES:
            self.assertTrue(_same(results[False][name], results[True][name]), name)
        for name in FRONTAL_NAMES + OBLIQUE_NAMES + ["cheek.yaw"]:
            self.assertTrue(math.isnan(results[True][name]), name)
        self.assertEqual(faces[True].calls[:-1], [])  # the pose line only
        self.assertEqual(
            faces[True].column, ["cheek yaw nan (model 0.0) pitch 0.1 (flagged)"]
        )
        self.assertEqual(faces[False].calls, [])

        # Handed over through frontal_cheek() with the right canthus NaN (so its
        # foot, reference points and zygion too): those points are skipped, not
        # rounded, the left side is drawn, and the values are render=False's.
        fc = cheek.frontal_constructions(Fixtures.frontal().landmarks)
        nan2 = np.array([math.nan, math.nan])
        broken = dataclasses.replace(
            fc,
            canthus_r=nan2,
            foot_r=nan2,
            cheek_ref_r=nan2,
            cheek_ref_r_px=math.nan,
            submalar_ref_r=nan2,
            zygion_r=nan2,
            zygion_r_px=math.nan,
        )
        results, faces = {}, {}
        for render in (False, True):
            face = Fixtures.frontal()
            face.frontal_cheek = lambda: broken
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results[render] = AnalyzeCheekMeasures().calc(face, render=render)
            self.assertEqual(_runtime_warnings(caught), [])
            faces[render] = face
        for name in EXPECTED_NAMES:
            self.assertTrue(_same(results[False][name], results[True][name]), name)
        for name in ("zy.r", "zy.w", "zy.ratio", "cr.r.d"):
            self.assertTrue(math.isnan(results[True][name]), name)
        for name in ("zy.l", "cr.l.d", "cheek.yaw"):
            self.assertFalse(math.isnan(results[True][name]), name)
        face = faces[True]
        # No level line (its right end is NaN) and one vertical; the left
        # zygion's rim and dot and the left side's two rings; three labels.
        self.assertEqual(len(face.kinds("line")), 1)
        self.assertEqual(len(face.kinds("circle")), 4)
        self.assertEqual(face.labels(), ["zygion", "cheek ref", "submalar ref"])
        for point in face.points:
            self.assertTrue(all(type(v) is int for v in point), point)
        self.assertEqual(len(face.column), 3)
        self.assertEqual(faces[False].calls, [])


class TestQuarter(unittest.TestCase):
    """The oblique module on the yaw45 render's traced outline."""

    def setUp(self):
        self.measure = AnalyzeCheekMeasures()
        self.face = Fixtures.quarter()
        self.result = self.measure.calc(self.face, render=True)

    def test_far_side_only(self):
        r = self.result
        self.assertTrue(self.face.far_left)
        self.assertEqual(r["cheek.far"], 1.0)
        for name in OBLIQUE_NAMES:
            if ".r." in name:
                self.assertTrue(math.isnan(r[name]), name)
            else:
                self.assertFalse(math.isnan(r[name]), name)
        for name in FRONTAL_NAMES:
            self.assertTrue(math.isnan(r[name]), name)

    def test_values_follow_observed_points(self):
        face = self.face
        ob = co.observed_points(
            face.landmarks,
            face.cheek_outline_x,
            face.cheek_outline_y,
            face.cheek_outline_src,
            True,
        )
        r = self.result
        c = coords(face)  # the Landmarks measure's Cheek Landmarks item
        self.assertEqual((c["apex.l.x"], c["apex.l.y"]), tuple(ob.apex))
        self.assertEqual(r["apex.l.h"], ob.apex_h)
        self.assertEqual(r["apex.l.dev"], ob.apex_dev)
        self.assertEqual((c["sub.l.x"], c["sub.l.y"]), tuple(ob.submalar))
        self.assertEqual(r["sub.l.h"], ob.submalar_h)
        self.assertEqual(r["sub.l.depth"], ob.submalar_depth)
        self.assertEqual(r["sub.l.depth.px"], ob.submalar_depth_px)
        self.assertEqual(r["sub.l.present"], ob.submalar_present)
        self.assertEqual(r["cheek.mask"], ob.mask_fraction)
        self.assertEqual(r["cheek.nose.gap"], ob.nose_gap)
        self.assertEqual(r["cheek.scale"], ob.scale)

    def test_known_facts_of_yaw45(self):
        r = self.result
        self.assertAlmostEqual(r["cheek.yaw"], 45.0, delta=0.1)
        self.assertAlmostEqual(r["cheek.yaw.model"], 45.0, delta=0.01)
        self.assertEqual(r["cheek.pitch"], 4.45)
        self.assertEqual(r["cheek.pitch.ok"], 1.0)
        self.assertEqual(r["apex.l.prior"], 1.0)
        self.assertAlmostEqual(r["apex.l.h"], co.APEX_H_PRIOR, delta=0.005)
        self.assertEqual(r["apex.l.near_nose"], 0.0)
        self.assertEqual(r["cheek.nose"], 0.0)
        self.assertGreater(r["cheek.nose.gap"], 0.04)
        self.assertEqual(r["cheek.mask"], 1.0)
        self.assertEqual(r["apex.ok"], 1.0)
        self.assertEqual(r["sub.l.present"], 0.0)
        self.assertLess(r["sub.l.depth"], co.SUBMALAR_MIN_DEPTH)
        self.assertAlmostEqual(r["sub.l.depth.px"], r["sub.l.depth"] * r["cheek.scale"])
        self.assertEqual(r["sub.ok"], 1.0)
        self.assertGreater(r["cheek.scale"], 200.0)

    def test_mirrored_turn_near_nose(self):
        # On the plain -45 render the found apex lies within 0.02 of the nose
        # tip's h, so near_nose folds into apex.ok, and the model's calibrated
        # yaw reads -44.8 (raw -25.63, less the offset).
        face = Fixtures.quarter("yaw-45")
        self.assertFalse(face.far_left)
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(result["cheek.far"], 0.0)
        self.assertAlmostEqual(result["cheek.yaw"], -45.0, delta=0.2)
        self.assertAlmostEqual(result["cheek.yaw.model"], -44.77, delta=0.01)
        self.assertEqual(result["apex.r.near_nose"], 1.0)
        self.assertEqual(result["apex.r.prior"], 0.0)
        self.assertEqual(result["apex.ok"], 0.0)
        self.assertEqual(result["sub.ok"], 0.0)
        self.assertFalse(math.isnan(result["sub.r.depth"]))
        for name in OBLIQUE_NAMES:
            if ".l." in name:
                self.assertTrue(math.isnan(result[name]), name)
        self.assertEqual(face.labels()[0], "malar apex (nose?)")
        self.assertEqual(face.column[0][-8:], " (nose?)")
        self.assertTrue(face.column[1].endswith(" (absent) (flagged)"))
        self.assertTrue(face.column[2].endswith(" (flagged)"))
        self.assertEqual(
            face.column[3:], flagged("the apex is level with the nose tip")
        )
        # The right cheek's labels go left of the point.
        apex_x = face.kinds("circle")[0][1][0]
        self.assertLess(face.kinds("text")[0][1][0], apex_x)
        # The far-right outline's level ticks point to the image right (inward).
        for call in face.kinds("line")[:3]:
            self.assertGreater(call[2][0], call[1][0])

    def test_hollow_render(self):
        face = Fixtures.quarter("yaw45_hollow3")
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(result["sub.l.present"], 1.0)
        self.assertGreaterEqual(result["sub.l.depth"], co.SUBMALAR_MIN_DEPTH)
        self.assertEqual(result["apex.ok"], 1.0)
        self.assertEqual(result["sub.ok"], 1.0)
        circles = face.kinds("circle")
        self.assertEqual(circles[-1][2:], (mc.GREEN, 7, -1))  # a filled dot
        self.assertEqual(face.labels()[-1], "submalar point")
        self.assertTrue(face.column[1].startswith("sub.l depth="))
        self.assertTrue(face.column[1].endswith(" (hollow)"))

    def test_drawing_calls(self):
        face = self.face
        for point in face.points:
            self.assertIsInstance(point, tuple)
            self.assertEqual(len(point), 2)
            for v in point:
                self.assertIs(type(v), int, point)
        polylines = face.kinds("polyline")
        self.assertEqual(len(polylines), 1)  # every band row from the mask
        self.assertEqual(polylines[0][2:], (mc.CYAN, 4))
        rows = [p[1] for p in polylines[0][1]]
        self.assertEqual(rows, sorted(rows))
        self.assertGreater(len(rows), 100)
        lines = face.kinds("line")
        self.assertEqual(len(lines), 4)
        for call in lines[:3]:  # the canthus, stomion and chin levels, inward
            self.assertEqual((call[3], call[4]), (mc.ORANGE, 1))
            self.assertLess(call[2][0], call[1][0])
            length = math.dist(call[1], call[2])
            self.assertAlmostEqual(length, 0.25 * self.result["cheek.scale"], delta=1.5)
        self.assertEqual(lines[0][1], polylines[0][1][0])  # from the outline
        self.assertEqual(lines[1][1], polylines[0][1][-1])
        self.assertEqual(lines[2][1], tuple(face.landmarks[16]))  # the chin
        chord = lines[3]
        self.assertEqual((chord[3], chord[4]), (mc.ORANGE, 2))
        self.assertEqual((chord[1], chord[2]), (lines[0][1], lines[1][1]))
        circles = face.kinds("circle")
        c = coords(face)
        apex = mc._pt((c["apex.l.x"], c["apex.l.y"]))
        self.assertEqual(circles[0], ("circle", apex, mc.WHITE, 9, -1))
        self.assertEqual(circles[1], ("circle", apex, mc.RED, 7, -1))
        # The absent submalar point is reported but not drawn: on a convex
        # cheek it sits beside a chord end, so a ring there marked nothing.
        self.assertEqual(self.result["sub.l.present"], 0.0)
        self.assertFalse(math.isnan(c["sub.l.x"]))
        self.assertEqual(len(circles), 2)
        self.assertEqual(face.labels(), ["malar apex (prior)"])
        texts = [call for call in face.kinds("text") if call[3] is not None]
        self.assertEqual(len(texts), 1)
        # Away from the face, beside the dot and centered on its row.
        self.assertEqual(texts[0][1], (apex[0] + 14, apex[1] + 10))
        self.assertEqual(texts[0][3], mc.RED)

    def test_text_column(self):
        self.assertEqual(
            self.face.column,
            [
                "apex.l h=0.79 dev=0.016 (prior)",
                "sub.l depth=0.001 (absent)",
                "cheek yaw 45.0 (model 45.0) pitch 4.5",
            ],
        )
        self.assertLessEqual(len(self.face.column), 5)

    def test_items_off_draw_nothing(self):
        # The frontal items stay on and draw nothing on a quarter face.
        measure = only(ZYGION, REFERENCE)
        face = Fixtures.quarter()
        result = measure.calc(face, render=True)
        self.assertEqual(face.calls, [])
        self.assertEqual(face.column, [])
        for name in EXPECTED_NAMES:
            self.assertTrue(_same(result[name], self.result[name]), name)

    def test_each_item_gates_exactly_its_own_drawing(self):
        """Each item alone draws its own share of the full drawing, in the same
        calls, and writes its own column line; the values never change."""
        full = self.face.calls
        outline = [c for c in full if c[0] in ("polyline", "line")]
        apex = [c for c in full if c[0] == "circle" or c[0] == "text" and c[3]]
        expected = {
            ZYGION: ([], []),
            REFERENCE: ([], []),
            OUTLINE: (outline, []),
            APEX: (apex, [self.face.column[0]]),
            SUBMALAR: ([], [self.face.column[1]]),  # absent: no dot, a line
            HEAD_TURN: ([], [self.face.column[2]]),
        }
        for name in ITEM_NAMES:
            with self.subTest(item=name):
                face = Fixtures.quarter()
                result = only(name).calc(face, render=True)
                drawn = [c for c in face.calls if c[0] != "text" or c[3] is not None]
                calls, column = expected[name]
                self.assertEqual(sorted(map(repr, drawn)), sorted(map(repr, calls)))
                self.assertEqual(face.column, column)
                self.assertEqual(list(result), EXPECTED_NAMES)
                for output in EXPECTED_NAMES:
                    self.assertTrue(_same(result[output], self.result[output]), output)
        # A present hollow: the submalar item alone draws its dot and label.
        face = Fixtures.quarter("yaw45_hollow3")
        only(SUBMALAR).calc(face, render=True)
        self.assertEqual([c[0] for c in face.calls], ["circle", "text", "text"])
        self.assertEqual(face.calls[0][2:], (mc.GREEN, 7, -1))
        self.assertEqual(face.labels(), ["submalar point"])
        self.assertEqual(len(face.column), 1)
        self.assertTrue(face.column[0].endswith(" (hollow)"))

    def test_render_false_draws_nothing(self):
        face = Fixtures.quarter()
        result = AnalyzeCheekMeasures().calc(face, render=False)
        self.assertEqual(face.calls, [])
        for name in EXPECTED_NAMES:
            self.assertTrue(_same(result[name], self.result[name]), name)

    def test_outline_colors_by_source(self):
        # Rows of every source: contour, nose and edge rows recolor their runs.
        face = Fixtures.quarter()
        src = np.array(face.cheek_outline_src, dtype=np.uint8)
        rows = face.cheek_outline_y
        canthus_row = face.landmarks[72][1]
        first = int(np.flatnonzero(rows >= canthus_row + 10)[0])
        src[first : first + 10] = co.OUTLINE_SRC_CONTOUR
        src[first + 20 : first + 30] = co.OUTLINE_SRC_EDGE
        src[first + 40 : first + 45] = co.OUTLINE_SRC_NOSE
        face.cheek_outline_src = src
        result = AnalyzeCheekMeasures().calc(face, render=True)
        polylines = face.kinds("polyline")
        colors = [call[2] for call in polylines]
        self.assertEqual(
            colors,
            [mc.CYAN, mc.GRAY, mc.CYAN, mc.YELLOW, mc.CYAN, mc.MAGENTA, mc.CYAN],
        )
        for call in polylines:
            self.assertEqual(call[3], 4)
        # The runs join: each starts where the previous one ended, and together
        # they cover every band row once.
        for before, after in zip(polylines, polylines[1:]):
            self.assertEqual(before[1][-1], after[1][0])
        total = sum(len(call[1]) - 1 for call in polylines) + 1
        self.assertEqual(
            total, len(polylines[0][1]) + sum(len(c[1]) - 1 for c in polylines[1:])
        )
        band_rows = sum(len(call[1]) - 1 for call in polylines) + 1
        self.assertGreater(band_rows, 100)
        # Nose rows set cheek.nose; the apex is then gray and flagged.
        self.assertEqual(result["cheek.nose"], 1.0)
        self.assertEqual(result["apex.ok"], 0.0)
        self.assertEqual(face.kinds("circle")[1][2], mc.GRAY)
        self.assertLess(result["cheek.mask"], 1.0)

    def test_contour_fallback(self):
        # The app's sketch, no outline arrays: the contour stands in, so the
        # band is all gray, no row came from the mask, the hollow is unknown,
        # and the apex is the prior (no local maximum is read on contour rows:
        # a bulge of SPIGA's contour polyline is a landmark, not an observation).
        face = Fixtures.sketch()
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(set(result), set(EXPECTED_NAMES))
        self.assertEqual(result["cheek.far"], 0.0)
        self.assertEqual(result["cheek.mask"], 0.0)
        for name in ("apex.r.h", "apex.r.dev", "cheek.scale"):
            self.assertFalse(math.isnan(result[name]), name)
        c = coords(face)
        for name in ("apex.r.x", "apex.r.y"):
            self.assertFalse(math.isnan(c[name]), name)
        for name in ("sub.r.x", "sub.r.y"):
            self.assertTrue(math.isnan(c[name]), name)
        for name in OBLIQUE_NAMES:
            if ".l." in name:
                self.assertTrue(math.isnan(result[name]), name)
        for name in (
            "sub.r.h",
            "sub.r.depth",
            "sub.r.depth.px",
            "sub.r.present",
        ):
            self.assertTrue(math.isnan(result[name]), name)
        self.assertEqual(result["apex.ok"], 0.0)
        self.assertEqual(result["sub.ok"], 0.0)
        self.assertEqual(result["cheek.nose"], 0.0)
        self.assertAlmostEqual(result["cheek.yaw"], -31.9, delta=0.1)
        self.assertAlmostEqual(result["cheek.yaw.model"], -32.9, delta=0.1)
        polylines = face.kinds("polyline")
        self.assertEqual(len(polylines), 1)
        self.assertEqual(polylines[0][2], mc.GRAY)
        circles = face.kinds("circle")
        self.assertEqual(len(circles), 2)  # the apex only: no submalar drawing
        self.assertEqual(circles[1][2], mc.GRAY)
        self.assertEqual(result["apex.r.prior"], 1.0)
        self.assertAlmostEqual(result["apex.r.h"], co.APEX_H_PRIOR, delta=0.005)
        self.assertEqual(face.labels(), ["malar apex (prior)"])
        self.assertTrue(face.column[0].startswith("apex.r h=0.79 dev="))
        self.assertTrue(face.column[0].endswith(" (prior)"))
        self.assertTrue(face.column[1].endswith(" (flagged)"))
        self.assertEqual(
            face.column[2:],
            flagged(
                "not turned enough (31.9 degrees, needs 40 to 50)",
                "the cheek's edge is unclear (hair or background)",
                "no submalar depth (the edge is unclear below the apex)",
            ),
        )
        # The same through explicit arrays from contour_outline().
        face2 = Fixtures.sketch(outline=co.contour_outline(face.landmarks, False))
        result2 = AnalyzeCheekMeasures().calc(face2, render=False)
        for name in EXPECTED_NAMES:
            self.assertTrue(_same(result[name], result2[name]), name)

    def test_stand_in_rows_at_the_prior_never_make_a_hollow(self):
        """
        An intensity-edge run 20 px outside the seed on the five rows at
        Kaptein's height of the yaw45 outline (a hair strand within the walk's
        reach): the apex is the mask row beside it with the mask's own x, apex.ok
        stays 1, and the submalar point stays absent with the plain outline's
        depth, where a chord started on the run read every row below as a
        hollow (depth 0.058 H, sub.l.present 1, sub.ok 1).
        """
        landmarks, far_left, _, (xs, rows, src) = Fixtures.traced("yaw45")
        self.assertTrue(far_left)
        plain = AnalyzeCheekMeasures().calc(Fixtures.quarter(), render=False)
        seed, _, _ = co.contour_outline(np.array(landmarks, dtype=np.float64), True)
        sel = (rows >= 334) & (rows <= 338)
        xs2, src2 = xs.copy(), src.copy()
        xs2[sel] = (seed[sel] + 20.0).astype(np.float32)
        src2[sel] = co.OUTLINE_SRC_EDGE
        face = Fixtures.quarter(outline=(xs2, rows, src2))
        result = AnalyzeCheekMeasures().calc(face, render=False)
        plain_at, at = coords(Fixtures.quarter()), coords(face)
        self.assertEqual(plain_at["apex.l.y"], 336.0)
        self.assertEqual(at["apex.l.y"], 339.0)
        self.assertEqual(result["apex.l.prior"], 1.0)
        self.assertEqual(result["apex.ok"], 1.0)
        self.assertLess(abs(at["apex.l.x"] - plain_at["apex.l.x"]), 0.5)
        self.assertAlmostEqual(result["cheek.mask"], 0.9713, delta=0.001)
        self.assertEqual(result["sub.l.present"], 0.0)
        self.assertEqual(result["sub.ok"], 1.0)
        self.assertAlmostEqual(result["sub.l.depth"], plain["sub.l.depth"], delta=2e-4)
        self.assertEqual(at["sub.l.y"], plain_at["sub.l.y"])

    def test_oblique_cheek_accessor_is_read(self):
        # A face with oblique_cheek() hands over its own result: the yaw45 points
        # with a sentinel height, on a face that carries no outline at all.
        face = Fixtures.quarter()
        ob = co.observed_points(
            face.landmarks,
            face.cheek_outline_x,
            face.cheek_outline_y,
            face.cheek_outline_src,
            True,
        )
        sentinel = dataclasses.replace(ob, apex_h=0.5, apex=np.array([600.0, 300.0]))
        face = Fixtures.quarter(outline=(None, None, None))
        face.oblique_cheek = lambda: sentinel
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(result["apex.l.h"], 0.5)
        self.assertEqual(result["apex.ok"], 1.0)
        self.assertEqual(face.kinds("circle")[0][1], (600, 300))
        self.assertEqual(face.column[0], "apex.l h=0.50 dev=0.016 (prior)")
        # And one whose accessor has nothing gives NaN, whatever the outline.
        face = Fixtures.quarter()
        face.oblique_cheek = lambda: None
        result = AnalyzeCheekMeasures().calc(face, render=True)
        for name in OBLIQUE_NAMES:
            if name == "cheek.far":
                self.assertEqual(result[name], 1.0)
            else:
                self.assertTrue(math.isnan(result[name]), name)
        self.assertEqual({call[0] for call in face.calls}, {"text"})
        self.assertEqual(
            face.column,
            ["cheek yaw 45.0 (model 45.0) pitch 4.5 (flagged)"]
            + flagged("the far cheek could not be measured"),
        )

    def test_far_side_follows_the_face(self):
        """Observed points that name the other side are reported on the face's,
        with a warning, so cheek.far and the reported side never disagree."""
        face = Fixtures.quarter()
        ob = co.observed_points(
            face.landmarks,
            face.cheek_outline_x,
            face.cheek_outline_y,
            face.cheek_outline_src,
            True,
        )
        face.oblique_cheek = lambda: dataclasses.replace(ob, far_left=False)
        with self.assertLogs("dynaface.measures_cheek", level="WARNING") as logs:
            result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("subject's right, the face says the left", logs.output[0])
        self.assertEqual(result["cheek.far"], 1.0)
        for name in EXPECTED_NAMES:
            self.assertTrue(_same(result[name], self.result[name]), name)
        self.assertEqual(face.column, self.face.column)
        self.assertEqual(face.calls, self.face.calls)
        # Agreeing sides log nothing.
        face = Fixtures.quarter()
        face.oblique_cheek = lambda: ob
        with self.assertNoLogs("dynaface.measures_cheek", level="WARNING"):
            result = AnalyzeCheekMeasures().calc(face, render=True)
        for name in EXPECTED_NAMES:
            self.assertTrue(_same(result[name], self.result[name]), name)
        self.assertEqual(face.calls, self.face.calls)

    def test_pitch_25_fails_only_pitch_ok(self):
        face = Fixtures.quarter(pitch=25.0)
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(result["cheek.pitch.ok"], 0.0)
        for name in EXPECTED_NAMES:
            if name in ("cheek.pitch", "cheek.pitch.ok"):
                continue
            self.assertTrue(_same(result[name], self.result[name]), name)
        self.assertEqual(result["apex.ok"], 1.0)
        self.assertEqual(
            face.column[-2:],
            ["cheek yaw 45.0 (model 45.0) pitch 25.0 (flagged)"]
            + flagged("head tipped up or down (pitch 25.0, needs within 20)"),
        )

    def test_nan_yaw_fails_every_gate(self):
        face = Fixtures.quarter(yaw=math.nan, pitch=math.nan)
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertTrue(math.isnan(result["cheek.yaw.model"]))
        self.assertAlmostEqual(result["cheek.yaw"], 45.0, delta=0.1)
        flags = [result[name] for name in OK_FLAGS if not math.isnan(result[name])]
        self.assertEqual(len(flags), 3)  # apex.ok, sub.ok, cheek.pitch.ok
        self.assertEqual(set(flags), {0.0})
        self.assertFalse(math.isnan(result["apex.l.h"]))  # still reported
        self.assertEqual(face.kinds("circle")[1][2], mc.GRAY)

    def test_yaw_gate(self):
        # The landmark yaw is what the gate reads: a model yaw 6 degrees off
        # fails the agreement, one 4 degrees off passes.
        face = Fixtures.quarter(yaw=cheek.YAW_CALIBRATION[-3][0])  # reads 55 degrees
        result = AnalyzeCheekMeasures().calc(face, render=False)
        self.assertAlmostEqual(result["cheek.yaw.model"], 55.0, delta=0.01)
        self.assertEqual(result["apex.ok"], 0.0)
        self.assertEqual(result["sub.ok"], 0.0)
        face = Fixtures.quarter(yaw=cheek.YAW_CALIBRATION[-5][0])  # reads 48 degrees
        result = AnalyzeCheekMeasures().calc(face, render=False)
        self.assertAlmostEqual(result["cheek.yaw.model"], 48.0, delta=0.01)
        self.assertEqual(result["apex.ok"], 1.0)

    def test_submalar_line_never_reads_negative_zero(self):
        # A convex cheek's depth just below zero reads 0.000 in the text column.
        ok = {"sub.ok": 1.0}
        ob = mock.Mock(submalar_depth=-2.6e-05, submalar_present=0.0)
        self.assertEqual(
            AnalyzeCheekMeasures._column_submalar(ob, True, ok),
            ["sub.l depth=0.000 (absent)"],
        )
        ob = mock.Mock(submalar_depth=-0.0123, submalar_present=0.0)
        self.assertEqual(
            AnalyzeCheekMeasures._column_submalar(ob, False, ok),
            ["sub.r depth=-0.012 (absent)"],
        )
        # A flagged one says so.
        self.assertEqual(
            AnalyzeCheekMeasures._column_submalar(ob, False, {"sub.ok": 0.0}),
            ["sub.r depth=-0.012 (absent) (flagged)"],
        )


class TestFlagReasons(unittest.TestCase):
    """
    A flagged three-quarter result says why in the text column, one "flagged:"
    line per reason after the item lines, under the items whose lines carry the
    flag; an unflagged result and a frontal face write none.
    """

    @staticmethod
    def reasons(face):
        return [line for line in face.column if line.startswith(mc.FLAG_PREFIX)]

    def test_unflagged_results_write_none(self):
        for face in (Fixtures.quarter(), Fixtures.frontal()):
            AnalyzeCheekMeasures().calc(face, render=True)
            self.assertEqual(self.reasons(face), [])
        face = Fixtures.quarter("yaw50")
        AnalyzeCheekMeasures().calc(face, render=False)
        self.assertEqual(face.column, [])  # nothing written without rendering

    def test_turned_too_far_with_the_nose_on_the_outline(self):
        # The 50 degree render, like a photo turned past the window: the turn
        # just past 50 and the nose on the far outline, whose rows also leave
        # no submalar depth. The nose is named, not an unclear edge.
        face = Fixtures.quarter("yaw50")
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(result["apex.ok"], 0.0)
        self.assertEqual(result["cheek.nose"], 1.0)
        self.assertGreater(abs(result["cheek.yaw"]), 50.0)
        self.assertEqual(
            self.reasons(face),
            flagged(
                "turned too far (just over 50 degrees, needs 40 to 50)",
                "the nose hides the far cheek",
                "no submalar depth (the nose covers the cheek below the apex)",
            ),
        )
        self.assertEqual(face.column[-3:], self.reasons(face))  # after the others

    def test_the_turn_estimates_differ(self):
        face = Fixtures.quarter(yaw=cheek.YAW_CALIBRATION[-3][0])  # reads 55
        result = AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(result["apex.ok"], 0.0)
        self.assertEqual(
            self.reasons(face),
            flagged("the turn estimates differ (45.0 and 55.0 degrees)"),
        )

    def test_reasons_follow_the_items(self):
        expected = {
            mc.ITEM_OUTLINE: [],
            mc.ITEM_APEX: flagged(
                "turned too far (just over 50 degrees, needs 40 to 50)",
                "the nose hides the far cheek",
            ),
            mc.ITEM_SUBMALAR: flagged(
                "no submalar depth (the nose covers the cheek below the apex)"
            ),
            mc.ITEM_HEAD_TURN: flagged(
                "turned too far (just over 50 degrees, needs 40 to 50)",
                "the nose hides the far cheek",
            ),
        }
        for name, reasons in expected.items():
            with self.subTest(item=name):
                measure = AnalyzeCheekMeasures()
                for item in mc.ITEM_NAMES:
                    measure.set_item_enabled(item, item == name)
                face = Fixtures.quarter("yaw50")
                measure.calc(face, render=True)
                self.assertEqual(self.reasons(face), reasons)

    def test_every_failed_gate_is_named(self):
        # Each apex gate alone, on the observed points of the plain 45 render.
        face = Fixtures.quarter()
        ob = co.observed_points(
            face.landmarks,
            face.cheek_outline_x,
            face.cheek_outline_y,
            face.cheek_outline_src,
            True,
        )
        cases = {
            "the nose hides the far cheek": dict(nose_on=True),
            "the apex is level with the nose tip": dict(apex_near_nose=True),
            "the cheek's edge is unclear (hair or background)": dict(window_mask=0.5),
        }
        for reason, change in cases.items():
            with self.subTest(reason=reason):
                changed = dataclasses.replace(ob, **change)
                self.assertEqual(mc._apex_reasons(changed, 45.0, 45.0, True), [reason])
        self.assertEqual(mc._apex_reasons(ob, 45.0, 45.0, True), [])
        self.assertEqual(
            mc._apex_reasons(ob, -38.2, -38.0, True),
            ["not turned enough (38.2 degrees, needs 40 to 50)"],
        )
        self.assertEqual(
            mc._apex_reasons(ob, math.nan, 45.0, False), ["the turn is unknown"]
        )
        self.assertEqual(
            mc._apex_reasons(ob, 45.0, math.nan, False), ["the model's turn is unknown"]
        )
        # The nose rows explain the window, so no unclear edge beside the nose.
        both = dataclasses.replace(ob, nose_on=True, window_mask=0.5)
        self.assertEqual(
            mc._apex_reasons(both, 45.0, 45.0, True), ["the nose hides the far cheek"]
        )

    def test_values_on_the_limit_read_past_it(self):
        self.assertEqual(mc._beyond(50.2, 50.0), "50.2")
        self.assertEqual(mc._beyond(50.04, 50.0), "just over 50")
        self.assertEqual(mc._beyond(39.96, 40.0), "just under 40")
        self.assertEqual(mc._beyond(-20.03, 20.0), "just under -20")
        self.assertEqual(mc._beyond(25.0, 20.0), "25.0")


class TestLabelsAtTheEdge(unittest.TestCase):
    """
    A label stays on its point's outside, never crossing the face: one that would
    cross the image's edge there is clamped to it and raised above its point.
    """

    @staticmethod
    def _labels(face):
        """Each point label as (x, y, width, text), in drawing order."""
        return [
            (call[1][0], call[1][1], 12 * len(call[2]), call[2])
            for call in face.kinds("text")
            if call[3] is not None
        ]

    def test_far_left_apex_near_the_right_edge(self):
        # The yaw45 render's far cheek is the subject's left, so its labels go
        # right, beside the apex; with the image cut to 20 px past the apex the
        # label is clamped to the right edge and raised above the apex, still
        # off the face rather than across it.
        wide = Fixtures.quarter()
        AnalyzeCheekMeasures().calc(wide, render=True)
        result = coords(wide)
        apex_x, apex_y = mc._pt((result["apex.l.x"], result["apex.l.y"]))
        self.assertTrue(wide.far_left)
        self.assertEqual(self._labels(wide)[0][:2], (apex_x + 14, apex_y + 10))
        face = Fixtures.quarter(width=apex_x + 20)
        AnalyzeCheekMeasures().calc(face, render=True)
        self.assertEqual(coords(face)["apex.l.x"], result["apex.l.x"])
        labels = self._labels(face)
        self.assertEqual([t for _, _, _, t in labels], face.labels())
        self.assertEqual(len(labels), 1)  # the apex alone: no hollow, no label
        for x, y, width, text in labels:
            self.assertGreaterEqual(x, 0, text)
            self.assertLessEqual(x + width, face.width, text)
        width = 12 * len("malar apex (prior)")
        self.assertEqual(labels[0][:2], (face.width - width, apex_y - 22))

    def test_far_right_apex_near_the_left_edge(self):
        # The yaw-45 render's far cheek is the subject's right, so its labels go
        # left; with the landmarks and outline shifted so the apex sits 20 px
        # from the left edge they flip right and stay inside.
        wide = Fixtures.quarter("yaw-45")
        AnalyzeCheekMeasures().calc(wide, render=True)
        result = coords(wide)
        self.assertFalse(wide.far_left)
        apex_x = mc._pt((result["apex.r.x"], 0))[0]
        self.assertLess(self._labels(wide)[0][0], apex_x)
        shift = apex_x - 20
        landmarks, far_left, raw, (xs, rows, src) = Fixtures.traced("yaw-45")
        face = Fixtures.quarter(
            "yaw-45",
            outline=(xs - shift, rows, src),
        )
        face.landmarks = [(x - shift, y) for x, y in landmarks]
        AnalyzeCheekMeasures().calc(face, render=True)
        shifted = coords(face)
        self.assertAlmostEqual(shifted["apex.r.x"], result["apex.r.x"] - shift)
        self.assertEqual(mc._pt((shifted["apex.r.x"], 0))[0], 20)
        labels = self._labels(face)
        self.assertEqual(len(labels), 1)
        for x, y, width, text in labels:
            self.assertGreaterEqual(x, 0, text)
            self.assertLessEqual(x + width, face.width, text)
        apex_y = mc._pt((0, shifted["apex.r.y"]))[1]
        self.assertEqual(labels[0][:2], (0, apex_y - 22))

    def test_frontal_labels_fit_beside_or_above(self):
        # The frontal labels fit on their outside at the image's width: the
        # zygion's above the dot, the reference points' beside the ring; none
        # comes within LABEL_GAP of another or of a dot.
        face = Fixtures.frontal()
        AnalyzeCheekMeasures().calc(face, render=True)
        labels = self._labels(face)
        self.assertEqual(labels[0][:2], (240 - 12 * len("zygion") - 14, 558 - 22))
        self.assertEqual(labels[1][:2], (778 + 14, 558 - 22))
        boxes = [(x, y - 20, x + width, y + 5) for x, y, width, _ in labels]
        boxes += [mc._dot_box(c[1], c[3]) for c in face.kinds("circle")]
        for x, y, width, text in labels:
            self.assertGreaterEqual(x, 0, text)
            self.assertLessEqual(x + width, face.width, text)
        for i, a in enumerate(boxes[: len(labels)]):
            for b in boxes[i + 1 :]:
                self.assertFalse(mc._overlaps(a, b), (labels[i][3], b))

    def test_clamped_on_its_outside(self):
        # On an image narrower than the label plus its offset, the label is
        # clamped to the image's edge on its outside and raised above its point.
        face = MockFace([], width=100)
        AnalyzeCheekMeasures._label(face, (50, 40), "malar apex", mc.RED, False)
        AnalyzeCheekMeasures._label(face, (50, 40), "malar apex", mc.RED, True)
        self.assertEqual(
            face.kinds("text"),
            [
                ("text", (0, 18), "malar apex", mc.RED, 1, 1),
                ("text", (0, 18), "malar apex", mc.RED, 1, 1),
            ],
        )
        # Room on the right but not the left: a label that belongs left stays
        # left, clamped to x = 0, rather than crossing to the right (over the
        # face).
        face = MockFace([], width=200)
        AnalyzeCheekMeasures._label(face, (50, 40), "malar apex", mc.RED, True)
        self.assertEqual(face.kinds("text")[0][1], (0, 18))
        # With room, beside the point, centered on its row.
        face = MockFace([], width=200)
        AnalyzeCheekMeasures._label(face, (50, 40), "malar apex", mc.RED, False)
        self.assertEqual(face.kinds("text")[0][1], (64, 50))


class TestPlaceLabel(unittest.TestCase):
    """place_label(): beside or above the point, and off what is drawn."""

    SIZE = (120, 20, 5)  # width, height above the baseline, depth below it

    def test_beside_and_above(self):
        self.assertEqual(
            mc.place_label((300, 200), self.SIZE, False, 1024, []), (314, 210)
        )
        self.assertEqual(
            mc.place_label((300, 200), self.SIZE, True, 1024, []), (166, 210)
        )
        self.assertEqual(
            mc.place_label((300, 200), self.SIZE, False, 1024, [], beside=False),
            (314, 178),
        )

    def test_taken_rows_move_the_label(self):
        # Its own row taken (a label already in its box): one line down (20 + 5
        # + LABEL_GAP = 29 px); that taken too: one line up; and the box is
        # added to taken.
        own = (314, 190, 434, 215)
        taken = [own]
        self.assertEqual(
            mc.place_label((300, 200), self.SIZE, False, 1024, taken), (314, 239)
        )
        self.assertEqual(taken[-1], (314, 219, 434, 244))
        taken = [own, (314, 219, 434, 244)]
        self.assertEqual(
            mc.place_label((300, 200), self.SIZE, False, 1024, taken), (314, 181)
        )
        # Every row taken: it stays on its own row.
        taken = [(0, 0, 1024, 1024)]
        self.assertEqual(
            mc.place_label((300, 200), self.SIZE, False, 1024, taken), (314, 210)
        )

    def test_a_dot_beside_its_label_is_clear(self):
        # A label beside its own dot (rim 9 px) keeps LABEL_GAP from it.
        taken = [mc._dot_box((300, 200), mc.DOT_RIM)]
        self.assertEqual(
            mc.place_label((300, 200), self.SIZE, False, 1024, taken), (314, 210)
        )
        self.assertEqual(
            mc.place_label((300, 200), self.SIZE, True, 1024, taken), (166, 210)
        )


class TestLateral(unittest.TestCase):
    def test_no_outputs_and_no_drawing(self):
        measure = AnalyzeCheekMeasures()
        for render in (False, True):
            face = Fixtures.lateral()
            self.assertEqual(measure.calc(face, render=render), {})
            self.assertEqual(face.calls, [])
            self.assertEqual(face.column, [])


class TestDrawCheekLandmarks(unittest.TestCase):
    """
    AnalyzeFace.draw_cheek_landmarks() on real faces restored without the
    models: a dot and a code for every cheek landmark that exists in the view,
    nothing for the rest, nor for a submalar point the outline shows no hollow
    at.
    """

    LATERAL_LANDMARKS = np.array(
        [[300, 200], [290, 300], [200, 450], [260, 520], [280, 640], [270, 760]]
    )
    SAGITTAL_Y = np.arange(1024, dtype=np.int32)
    SAGITTAL_X = (300 + SAGITTAL_Y % 50).astype(np.int32)

    @staticmethod
    def restore(landmarks, view):
        """A face from a state of these landmarks and view, on a blank image, so
        load_state() locates the cheek landmarks as it does for a saved frame."""
        face = facial.AnalyzeFace(measures=[])
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        state = [image, [0.0] * 6, [tuple(p) for p in landmarks], None, None, 0, view]
        face.load_state(state)
        return face

    @classmethod
    def frontal(cls):
        pipeline = Fixtures.images()["img4"]["pipeline"]
        landmarks = [tuple(int(v) for v in xy) for xy in pipeline["landmarks"]]
        return cls.restore(
            landmarks, {"pose": "frontal", "flipped": False, "lateral": False}
        )

    @classmethod
    def quarter(cls, name="yaw45"):
        landmarks, far_left, _, outline = Fixtures.traced(name)
        view = {
            "pose": "quarter",
            "flipped": False,
            "lateral": False,
            "far_left": far_left,
            "cheek_outline_x": outline[0],
            "cheek_outline_y": outline[1],
            "cheek_outline_src": outline[2],
        }
        return cls.restore(landmarks, view)

    @classmethod
    def lateral(cls):
        pipeline = Fixtures.images()["img4"]["pipeline"]
        landmarks = [tuple(int(v) for v in xy) for xy in pipeline["landmarks"]]
        view = {
            "pose": "lateral",
            "flipped": False,
            "lateral": True,
            "lateral_landmarks": cls.LATERAL_LANDMARKS,
            "sagittal_x": cls.SAGITTAL_X,
            "sagittal_y": cls.SAGITTAL_Y,
        }
        return cls.restore(landmarks, view)

    @staticmethod
    def drawn(face, **kwargs):
        """The dots and the texts draw_cheek_landmarks() draws, as recorded calls."""
        with mock.patch.object(
            facial.AnalyzeFace, "circle"
        ) as circle, mock.patch.object(facial.AnalyzeFace, "write_text") as text:
            face.draw_cheek_landmarks(**kwargs)
        return circle.call_args_list, text.call_args_list

    def test_codes_name_the_rows(self):
        self.assertEqual(len(cheek.CHEEK_LANDMARK_CODES), 10)
        self.assertEqual(
            len(cheek.CHEEK_LANDMARK_CODES), len(cheek.CHEEK_LANDMARK_NAMES)
        )
        # Each code is the prefix of that landmark's x and y outputs, the
        # Landmarks measure's Cheek Landmarks item.
        for code in cheek.CHEEK_LANDMARK_CODES:
            self.assertIn(code + ".x", COORD_NAMES)
            self.assertIn(code + ".y", COORD_NAMES)
        self.assertEqual(len(COORD_NAMES), 20)
        self.assertEqual(cheek.CHEEK_LANDMARK_CODES[cheek.CHEEK_LM_ZYGION_R], "zy.r")
        self.assertEqual(cheek.CHEEK_LANDMARK_CODES[cheek.CHEEK_LM_APEX_L], "apex.l")
        self.assertEqual(cheek.CHEEK_LANDMARK_CODES[cheek.CHEEK_LM_SUBMALAR_L], "sub.l")

    def test_frontal_face_draws_its_six_constructions(self):
        face = self.frontal()
        self.assertTrue(np.isfinite(face.cheek_landmarks[:6]).all())
        circles, texts = self.drawn(face, size=0.5)
        self.assertEqual(len(circles), 6)
        self.assertEqual(
            [call.args[1] for call in texts],
            ["zy.r", "zy.l", "cr.r", "cr.l", "sr.r", "sr.l"],
        )
        # The same dot size rule as draw_landmarks(), and the color given.
        radius = max(1, round(3 * 0.5))
        for call, point in zip(circles, face.cheek_landmarks[:6]):
            self.assertEqual(call.kwargs["radius"], radius)
            self.assertEqual(call.kwargs["color"], (255, 128, 0))
            self.assertEqual(call.args[0], (int(round(point[0])), int(round(point[1]))))
        # The code sits just past the dot, at the text size given.
        for call, circle in zip(texts, circles):
            x, y = circle.args[0]
            self.assertEqual(call.args[0], (x + radius + 1, y))
            self.assertEqual(call.kwargs["size"], 0.5)

    def test_options(self):
        face = self.frontal()
        circles, texts = self.drawn(face, names=False)
        self.assertEqual(len(circles), 6)
        self.assertEqual(texts, [])
        circles, _ = self.drawn(face, size=1.0, color=(1, 2, 3))
        self.assertEqual(circles[0].kwargs["radius"], 3)
        self.assertEqual(circles[0].kwargs["color"], (1, 2, 3))

    def test_quarter_face_draws_the_far_cheeks_points(self):
        # yaw45_hollow3 has a hollow under the apex: both points are drawn.
        face = self.quarter("yaw45_hollow3")
        self.assertTrue(face.quarter and face.far_left)
        self.assertEqual(face.oblique_cheek().submalar_present, 1.0)
        circles, texts = self.drawn(face)
        self.assertEqual(len(circles), 2)
        self.assertEqual([call.args[1] for call in texts], ["apex.l", "sub.l"])
        apex = face.cheek_landmarks[cheek.CHEEK_LM_APEX_L]
        sub = face.cheek_landmarks[cheek.CHEEK_LM_SUBMALAR_L]
        self.assertEqual(
            [call.args[0] for call in circles],
            [(int(round(p[0])), int(round(p[1]))) for p in (apex, sub)],
        )

    def test_absent_submalar_point_is_not_drawn(self):
        # yaw45 shows no hollow: its submalar point is still a cheek landmark,
        # and exported, but only the apex is drawn, as the cheek measure does.
        face = self.quarter()
        self.assertTrue(face.quarter and face.far_left)
        self.assertEqual(face.oblique_cheek().submalar_present, 0.0)
        self.assertTrue(
            np.isfinite(face.cheek_landmarks[cheek.CHEEK_LM_SUBMALAR_L]).all()
        )
        circles, texts = self.drawn(face)
        self.assertEqual(len(circles), 1)
        self.assertEqual([call.args[1] for call in texts], ["apex.l"])
        apex = face.cheek_landmarks[cheek.CHEEK_LM_APEX_L]
        self.assertEqual(circles[0].args[0], (int(round(apex[0])), int(round(apex[1]))))

    def test_nothing_without_a_point(self):
        lateral = self.lateral()
        self.assertTrue(np.isnan(lateral.cheek_landmarks).all())
        self.assertEqual(self.drawn(lateral), ([], []))
        no_face = facial.AnalyzeFace(measures=[])
        self.assertTrue(no_face.is_no_face())
        self.assertEqual(self.drawn(no_face), ([], []))

    def test_draws_on_the_image(self):
        face = self.frontal()
        before = face.render_img.copy()
        face.draw_cheek_landmarks()
        self.assertFalse(np.array_equal(before, face.render_img))


class TestCheekLandmarksItem(unittest.TestCase):
    """
    The Landmarks measure's "Cheek Landmarks" item on real faces restored without
    the models: its 20 outputs are face.cheek_landmarks (the coordinates the cheek
    measure used to report, the same values), NaN where absent, and it draws
    every cheek landmark that exists in the view, but a submalar point without a
    hollow, while it is enabled.
    """

    @staticmethod
    def sketch():
        """The app's three-quarter sketch restored as a quarter face without an
        outline: the contour stands in, so its submalar point is NaN."""
        raw = Fixtures.images()["quarter_sketch"]["raw"]
        landmarks = [tuple(int(v) for v in xy) for xy in raw["landmarks"]]
        view = {
            "pose": "quarter",
            "flipped": False,
            "lateral": False,
            "far_left": False,
            "cheek_outline_x": None,
            "cheek_outline_y": None,
            "cheek_outline_src": None,
        }
        return TestDrawCheekLandmarks.restore(landmarks, view)

    def check_values(self, face):
        result = AnalyzeLandmarks().calc(face, render=False)
        values = [result[name] for name in COORD_NAMES]
        for name, value, expected in zip(
            COORD_NAMES, values, face.cheek_landmarks.reshape(-1)
        ):
            self.assertIs(type(value), float, name)
            self.assertTrue(_same(value, float(expected)), name)
        return dict(zip(COORD_NAMES, values))

    def finite(self, values):
        return [name for name, value in values.items() if not math.isnan(value)]

    def test_frontal_face(self):
        face = TestDrawCheekLandmarks.frontal()
        values = self.check_values(face)
        self.assertEqual(self.finite(values), COORD_NAMES[:12])
        # The constructions' pinned places, as the cheek measure drew them.
        self.assertEqual(mc._pt((values["zy.r.x"], values["zy.r.y"])), (240, 558))
        self.assertEqual(mc._pt((values["zy.l.x"], values["zy.l.y"])), (778, 558))
        # The mock face (no cheek_landmarks) gets the same values through
        # cheek.locate().
        mock_values = coords(Fixtures.frontal())
        for name in COORD_NAMES:
            self.assertTrue(_same(mock_values[name], values[name]), name)

    def test_quarter_face(self):
        face = TestDrawCheekLandmarks.quarter()
        values = self.check_values(face)
        self.assertEqual(
            self.finite(values), ["apex.l.x", "apex.l.y", "sub.l.x", "sub.l.y"]
        )
        self.assertEqual(values["apex.l.y"], 336.0)
        mock_values = coords(Fixtures.quarter())
        for name in COORD_NAMES:
            self.assertTrue(_same(mock_values[name], values[name]), name)

    def test_absent_points_are_nan(self):
        sketch = self.sketch()
        self.assertTrue(sketch.quarter)
        self.assertEqual(
            self.finite(self.check_values(sketch)), ["apex.r.x", "apex.r.y"]
        )
        lateral = TestDrawCheekLandmarks.lateral()
        self.assertEqual(self.finite(self.check_values(lateral)), [])

    @staticmethod
    def drawn(face, measure=None, render=True):
        """The circles and texts the item draws: the 98 landmarks' drawing is
        patched out."""
        measure = measure or AnalyzeLandmarks()
        with mock.patch.object(
            facial.AnalyzeFace, "draw_landmarks"
        ) as landmarks, mock.patch.object(
            facial.AnalyzeFace, "circle"
        ) as circle, mock.patch.object(
            facial.AnalyzeFace, "write_text"
        ) as text:
            measure.calc(face, render=render)
        return landmarks.call_count, circle.call_args_list, text.call_args_list

    def test_drawing(self):
        face = TestDrawCheekLandmarks.frontal()
        landmarks, circles, texts = self.drawn(face)
        self.assertEqual(landmarks, 2)  # the 98, black under white
        self.assertEqual(texts, [])  # no labels
        self.assertEqual(len(circles), 12)
        # A black dot of size 1.0 under an orange one of size 0.5, for each.
        self.assertEqual(
            [(c.kwargs["color"], c.kwargs["radius"]) for c in circles],
            [((0, 0, 0), 3)] * 6 + [((255, 128, 0), 2)] * 6,
        )
        points = [(int(round(x)), int(round(y))) for x, y in face.cheek_landmarks[:6]]
        self.assertEqual([c.args[0] for c in circles], points * 2)

    def test_quarter_drawing_skips_an_absent_submalar_point(self):
        # Absent (yaw45): the apex alone; present (yaw45_hollow3): both. The
        # values are the same either way, the submalar point's included.
        for name, rows in (
            ("yaw45", [cheek.CHEEK_LM_APEX_L]),
            ("yaw45_hollow3", [cheek.CHEEK_LM_APEX_L, cheek.CHEEK_LM_SUBMALAR_L]),
        ):
            face = TestDrawCheekLandmarks.quarter(name)
            values = self.check_values(face)
            self.assertEqual(
                self.finite(values), ["apex.l.x", "apex.l.y", "sub.l.x", "sub.l.y"]
            )
            _, circles, texts = self.drawn(face)
            self.assertEqual(texts, [], name)
            points = [
                (int(round(x)), int(round(y))) for x, y in face.cheek_landmarks[rows]
            ]
            # A black dot under an orange one, for each point drawn.
            self.assertEqual([c.args[0] for c in circles], points * 2, name)

    def test_drawing_gated_by_the_item(self):
        face = TestDrawCheekLandmarks.frontal()
        measure = AnalyzeLandmarks()
        measure.set_item_enabled("Cheek Landmarks", False)
        landmarks, circles, texts = self.drawn(face, measure)
        self.assertEqual((landmarks, circles, texts), (2, [], []))
        self.assertEqual(self.drawn(face, render=False), (0, [], []))
        # Off, the values are still returned, and get_all_items() drops them.
        result = measure.calc(face, render=False)
        self.assertTrue(set(COORD_NAMES) <= set(result))
        listed = measures.get_all_items([measure])
        self.assertFalse(set(COORD_NAMES) & set(listed))
        self.assertEqual(len(listed), 196)

    def test_nothing_drawn_for_nan_rows(self):
        sketch = self.sketch()
        _, circles, _ = self.drawn(sketch)
        apex = sketch.cheek_landmarks[cheek.CHEEK_LM_APEX_R]
        point = (int(round(apex[0])), int(round(apex[1])))
        self.assertEqual([c.args[0] for c in circles], [point, point])
        quarter = TestDrawCheekLandmarks.quarter()
        self.assertEqual(len(self.drawn(quarter)[1]), 2)  # apex; submalar absent
        hollow = TestDrawCheekLandmarks.quarter("yaw45_hollow3")
        self.assertEqual(len(self.drawn(hollow)[1]), 4)  # apex and submalar
        self.assertEqual(self.drawn(TestDrawCheekLandmarks.lateral())[1], [])


if __name__ == "__main__":
    unittest.main()
