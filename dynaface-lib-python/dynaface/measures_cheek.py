"""The cheek measures: the frontal constructions and the oblique observed points,
in six items, each drawing one thing and carrying the numbers about it."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from dynaface import cheek, cheek_outline, util
from dynaface.const import Pose
from dynaface.measures_base import MeasureBase, MeasureItem, select_outputs

logger = logging.getLogger(__name__)

# The six items' names: each is a switch named for what it draws, standing for
# the outputs about it (MeasureItem's outputs).
ITEM_ZYGION = "Zygion width"
ITEM_REFERENCE = "Reference points"
ITEM_OUTLINE = "Far cheek outline"
ITEM_APEX = "Malar apex"
ITEM_SUBMALAR = "Submalar point"
ITEM_HEAD_TURN = "Head turn"

#: The items, in order: (name, the views it suits, its outputs in CSV order).
#: The frontal items' outputs are NaN on a quarter face and the quarter items'
#: on a frontal face (and always for the near side); the head turn's are numeric
#: in either view (cheek.yaw is NaN without a pupil frame).
ITEMS: Tuple[Tuple[str, Tuple[Pose, ...], Tuple[str, ...]], ...] = (
    (
        ITEM_ZYGION,
        (Pose.FRONTAL,),
        (
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
        ),
    ),
    (
        ITEM_REFERENCE,
        (Pose.FRONTAL,),
        ("cr.r.d", "cr.l.d", "cr.ok", "sr.ok"),
    ),
    (
        ITEM_OUTLINE,
        (Pose.QUARTER,),
        ("cheek.far", "cheek.mask", "cheek.nose", "cheek.nose.gap", "cheek.scale"),
    ),
    (
        ITEM_APEX,
        (Pose.QUARTER,),
        (
            "apex.r.h",
            "apex.l.h",
            "apex.r.dev",
            "apex.l.dev",
            "apex.r.prior",
            "apex.l.prior",
            "apex.r.near_nose",
            "apex.l.near_nose",
            "apex.ok",
        ),
    ),
    (
        ITEM_SUBMALAR,
        (Pose.QUARTER,),
        (
            "sub.r.h",
            "sub.l.h",
            "sub.r.depth",
            "sub.l.depth",
            "sub.r.depth.px",
            "sub.l.depth.px",
            "sub.r.present",
            "sub.l.present",
            "sub.ok",
        ),
    ),
    (
        ITEM_HEAD_TURN,
        (Pose.FRONTAL, Pose.QUARTER),
        ("cheek.yaw", "cheek.yaw.model", "cheek.pitch", "cheek.pitch.ok"),
    ),
)

#: The six items' names, in order.
ITEM_NAMES: Tuple[str, ...] = tuple(name for name, _, _ in ITEMS)

#: Every output, in CSV order: the items' outputs, item by item (41).
OUTPUTS: Tuple[str, ...] = tuple(name for _, _, outputs in ITEMS for name in outputs)

# The drawing's colors, RGB (render_img is RGB).
ORANGE = (255, 140, 0)  # the level lines, the verticals, the rings and the chord
BLUE = (31, 119, 180)  # the zygion dots and labels
CYAN = (0, 200, 255)  # outline rows from the mask
RED = (214, 39, 40)  # the malar apex
GREEN = (44, 160, 44)  # the submalar point
GRAY = (110, 110, 110)  # a flagged point; outline rows where the contour stood in
MAGENTA = (255, 0, 255)  # outline rows the nose joins
YELLOW = (255, 230, 0)  # outline rows where an intensity edge stood in for the mask
WHITE = (255, 255, 255)  # the rim of a filled dot

#: The far outline's color by row source (cheek_outline.OUTLINE_SRC_*).
OUTLINE_COLORS: Dict[int, Tuple[int, int, int]] = {
    cheek_outline.OUTLINE_SRC_CONTOUR: GRAY,
    cheek_outline.OUTLINE_SRC_MASK: CYAN,
    cheek_outline.OUTLINE_SRC_NOSE: MAGENTA,
    cheek_outline.OUTLINE_SRC_EDGE: YELLOW,
}

# The dots: a white rim of DOT_RIM px under a filled dot of DOT_RADIUS px; a
# reference point is an open ring of DOT_RADIUS px and RING_THICKNESS px.
DOT_RIM, DOT_RADIUS, RING_THICKNESS = 9, 7, 3
# A label sits LABEL_DX px outside its point, LABEL_DY px above it (inside it,
# or clamped, when the outside would cross the image's edge: see _label()).
LABEL_DX, LABEL_DY = 14, 22
# The zygion level line runs this far past each zygion, in IPD units.
LEVEL_OVERHANG_IPD = 0.15
# Each oblique level is drawn from the outline inward by this fraction of H.
OBLIQUE_LEVEL_LENGTH = 0.25
# The far outline's polyline, px; the level ticks are 1 px, the chord 2 px.
OUTLINE_THICKNESS = 4


def _pt(p: ArrayLike) -> Tuple[int, int]:
    """
    A point as a tuple of ints for OpenCV, which rejects a float array and even
    a float tuple.

    Args:
        p (ArrayLike): The point, (x, y) px.

    Returns:
        Tuple[int, int]: Its coordinates, each rounded to the nearest pixel.
    """
    q = np.asarray(p, dtype=np.float64).reshape(-1)
    return int(round(float(q[0]))), int(round(float(q[1])))


def _flag(value: bool) -> float:
    """A flag as 1.0 or 0.0."""
    return 1.0 if value else 0.0


def _finite(*points: ArrayLike) -> bool:
    """
    Whether every point can be drawn: all of their coordinates are finite, so
    _pt() can round them. A construction handed over with a NaN point is
    skipped rather than raised on, so calc(render=True) returns wherever
    calc(render=False) does.

    Args:
        *points (ArrayLike): The points, (x, y) px each.

    Returns:
        bool: True when every coordinate is finite.
    """
    return all(bool(np.isfinite(np.asarray(p, dtype=np.float64)).all()) for p in points)


def _apex_note(ob: cheek_outline.ObliqueCheek) -> str:
    """
    The apex label's suffix: " (prior)" when the prior stood in for a local
    maximum, " (nose?)" when its height is within APEX_NEAR_NOSE_H of the nose
    tip's, " (prior, nose?)" for both, "" otherwise.

    Args:
        ob (cheek_outline.ObliqueCheek): The observed points.

    Returns:
        str: The suffix.
    """
    notes = []
    if ob.apex_prior:
        notes.append("prior")
    if ob.apex_near_nose:
        notes.append("nose?")
    return f" ({', '.join(notes)})" if notes else ""


#: Each text column line that says why a three-quarter result is flagged
#: starts with this.
FLAG_PREFIX = "flagged: "


def _beyond(value: float, bound: float) -> str:
    """
    A value outside a limit, to one decimal place, or "just over 50" (or "just
    under 40", or "just under -20" for a negative value past -20) where one place
    would round it onto the limit, so the text never reads "50.0 degrees, needs
    40 to 50".

    Args:
        value (float): The value, such as a turn in degrees.
        bound (float): The limit it is compared with, positive.

    Returns:
        str: The value, signed as given.
    """
    text = f"{value:.1f}"
    if abs(float(text)) != bound:
        return text
    limit = math.copysign(bound, value)
    side = "over" if value > limit else "under"
    return f"just {side} {limit:g}"


def _submalar_reason(ob: cheek_outline.ObliqueCheek) -> str:
    """
    Why the submalar depth couldn't be read: too few of the rows from the apex
    down came from the mask. When nose rows (OUTLINE_SRC_NOSE) make up at least
    half of the rows that didn't, the nose is named instead of an unclear edge.

    Args:
        ob (cheek_outline.ObliqueCheek): The observed points.

    Returns:
        str: The phrase.
    """
    rows = np.asarray(ob.outline, dtype=np.float64)
    src = np.asarray(ob.outline_src)
    apex_y = float(np.asarray(ob.apex, dtype=np.float64)[1])
    if rows.size and src.size == len(rows) and math.isfinite(apex_y):
        below = src[rows[:, 1] >= apex_y]
        others = int(np.count_nonzero(below != cheek_outline.OUTLINE_SRC_MASK))
        nose = int(np.count_nonzero(below == cheek_outline.OUTLINE_SRC_NOSE))
        if nose and 2 * nose >= others:
            return "no submalar depth (the nose covers the cheek below the apex)"
    return "no submalar depth (the edge is unclear below the apex)"


def _apex_reasons(
    ob: cheek_outline.ObliqueCheek, yaw: float, yaw_model: float, agree: bool
) -> List[str]:
    """
    Why apex.ok is 0, as short phrases for the text column, in the order its
    gates are checked: the turn outside OBLIQUE_YAW +/- OBLIQUE_YAW_GATE, the
    two yaws more than YAW_AGREE_DEG apart, the nose on the far outline, the
    apex within APEX_NEAR_NOSE_H of the nose tip's height, and fewer than
    OUTLINE_MIN_MASK of the search window's rows from the mask. Every gate that
    failed has its phrase, so a flagged apex always has at least one, except
    that the window's rows are not blamed on an unclear edge when the nose is
    on the outline: its rows are the ones that didn't come from the mask.

    Args:
        ob (cheek_outline.ObliqueCheek): The observed points and their flags.
        yaw (float): cheek.yaw, degrees.
        yaw_model (float): cheek.yaw.model, degrees.
        agree (bool): Whether the two yaws agree.

    Returns:
        List[str]: One phrase per failed gate; empty when none failed.
    """
    reasons: List[str] = []
    low = cheek.OBLIQUE_YAW - cheek.OBLIQUE_YAW_GATE
    high = cheek.OBLIQUE_YAW + cheek.OBLIQUE_YAW_GATE
    window = f"needs {low:.0f} to {high:.0f}"
    turn = abs(yaw)
    if math.isnan(turn):
        reasons.append("the turn is unknown")
    elif turn > high:
        reasons.append(f"turned too far ({_beyond(turn, high)} degrees, {window})")
    elif turn < low:
        reasons.append(f"not turned enough ({_beyond(turn, low)} degrees, {window})")
    if not agree and not math.isnan(turn):
        if math.isnan(yaw_model):
            reasons.append("the model's turn is unknown")
        else:
            reasons.append(
                f"the turn estimates differ ({yaw:.1f} and {yaw_model:.1f} degrees)"
            )
    if ob.nose_on:
        reasons.append("the nose hides the far cheek")
    if ob.apex_near_nose:
        reasons.append("the apex is level with the nose tip")
    if ob.window_mask < cheek_outline.OUTLINE_MIN_MASK and not ob.nose_on:
        reasons.append("the cheek's edge is unclear (hair or background)")
    return reasons


class AnalyzeCheekMeasures(MeasureBase):
    """
    Cheek Measures: on a frontal face, the zygion width and the cheek and
    submalar reference points constructed from the landmarks; on a three-quarter
    (quarter) face, the far cheek's outline and the malar apex and submalar
    point observed on it; on both, the head turn the gates read. The cheek
    landmarks' coordinates themselves are the Landmarks measure's "Cheek
    Landmarks" item (cheek.CHEEK_LANDMARK_OUTPUTS), not outputs of this measure.
    Frontal distances are millimeters (face.pix2mm, measured on the landmarks
    being measured); oblique heights and depths are fractions of H, the
    chin-to-canthus distance (cheek.scale), because the pupil scale over-reads a
    turned head by about 1.4.

    Six items, in this order, each a group (MeasureItem's outputs): a switch named
    for the one thing it draws, standing for the numbers about it, and limited to
    the views where it means something (MeasureItem's views), so on a quarter face
    the frontal items are off after update_for_type() or MeasureSelection.apply()
    and the other way round. The values never depend on render or on which items
    are enabled, and every one of the 41 outputs is returned either way, in the
    items' order (OUTPUTS). For a profile, this measure returns no outputs ({}),
    like AnalyzeLateral on a frontal face.

    1. "Zygion width" (frontal), NaN on a quarter face:
        - `zy.r`, `zy.l`: Each constructed zygion's distance from the midline (the
          pupil-axis normal through the pupil midpoint), mm; `zy.w`: the
          bizygomatic width, their sum, mm; `zy.ratio`:
          util.symmetry_ratio(zy.r, zy.l), always reported; `zy.ratio.ok`: 1 when
          zy.ok is 1 and abs(cheek.yaw) <= 2.
        - `zy.r.clip`, `zy.l.clip`: 0 when the level crossed contour 0-4 / 32-28,
          1 when it passed above the top (the point is 0/32), 2 when it passed
          below the bottom (the point is 4/28).
        - `zy.r.ok`, `zy.l.ok`: 1 when that side isn't clipped and the yaw gate
          passes (abs(cheek.yaw) <= 5 with the agreement); `zy.ok`: 1 when both
          are, the gate of the width and the ratio.

        Draws the orange zygion level line, a blue zygion dot with a white rim at
        each end (gray when that side's ok flag is 0), each labeled "zygion",
        " (clipped)" appended, and the zygion line of the text column.

    2. "Reference points" (frontal), NaN on a quarter face:
        - `cr.r.d`, `cr.l.d`: The depth of each cheek reference point (Wilkinson's
          third) below the lateral canthus along the vertical, mm; `cr.ok`: 1
          when the yaw gate passes (per pair: the gate is the pose alone, and the
          foot always exists); `sr.ok`: the same gate, for the submalar
          reference points.

        Draws an orange vertical from each lateral canthus to its foot on the
        contour and an open orange ring for each reference point (gray when cr.ok
        is 0), labeled "cheek reference point" and "submalar reference point", and
        the reference line of the text column.

    3. "Far cheek outline" (quarter), NaN on a frontal face:
        - `cheek.far`: 0 when the far (measured) cheek is the subject's right, 1
          the left, on every quarter face, with or without a pupil frame;
          `cheek.mask`: the fraction of outline rows between the canthus and
          stomion levels that came from the mask; `cheek.nose`: 1 when the nose
          joins the far outline; `cheek.nose.gap`: the signed gap from the nose
          tip to the outline at the tip's row, positive inside the face, fraction
          of H; `cheek.scale`: H, the chin-to-far-canthus distance along the
          vertical through the far canthus, px.

        Draws the far outline from the canthus level to the stomion level as a 4 px
        polyline, cyan where the rows came from the mask, gray where the contour
        stood in, magenta on the rows the nose joins and yellow where an intensity
        edge stood in; the canthus, stomion and chin levels as 1 px orange ticks
        from the outline inward by 0.25 H; and the canthus-to-stomion chord as a
        2 px orange line, which also shows the span the submalar point was looked
        for in.

    4. "Malar apex" (quarter), NaN on a frontal face and for the near side:
        - `apex.r.h`, `apex.l.h`: The observed malar apex's Kaptein height ratio
          (chin level to the point over chin level to canthus level, read along
          the vertical at the point's x); `apex.r.dev`, `apex.l.dev`: its outward
          deviation from the canthus-to-stomion chord, fraction of H;
          `apex.r.prior`, `apex.l.prior`: 1 when no local maximum qualified and
          the point at h = 0.793 was used; `apex.r.near_nose`, `apex.l.near_nose`:
          1 when its h is within 0.02 of the nose tip's; `apex.ok`: 1 when
          abs(abs(cheek.yaw) - 45) <= 5 with the yaw agreement, cheek.nose is 0,
          near_nose is 0 and at least OUTLINE_MIN_MASK of the search window's
          rows came from the mask.

        Draws the apex as a red dot with a white rim (gray when apex.ok is 0),
        labeled "malar apex" plus " (prior)" or " (nose?)", and the apex line of
        the text column.

    5. "Submalar point" (quarter), NaN on a frontal face and for the near side:
        - `sub.r.h`, `sub.l.h`: The observed submalar point's height ratio, the
          deepest point below the apex down to the stomion level against the
          apex-to-stomion chord; `sub.r.depth`, `sub.l.depth`: its depth inward
          from that chord, fraction of H, positive inward, so a convex cheek
          reads about zero or slightly negative, NaN when fewer than
          OUTLINE_MIN_MASK of the rows below the apex came from the mask;
          `sub.r.depth.px`, `sub.l.depth.px`: the same in px; `sub.r.present`,
          `sub.l.present`: 1 when the depth reaches SUBMALAR_MIN_DEPTH, 0 below,
          NaN when the depth is; `sub.ok`: 1 when apex.ok is 1 and the depth
          isn't NaN.

        Draws the submalar point as a green dot labeled "submalar point" when
        present, nothing otherwise (an absent point sits wherever the outline
        comes nearest the chord, on a convex cheek beside one of the chord's
        ends, not a landmark), and the submalar line of the text column.

    6. "Head turn" (frontal and quarter), numeric in either view (`cheek.yaw` is
       NaN without a pupil frame):
        - `cheek.yaw`: The turn measured from the landmarks (cheek.landmark_yaw),
          degrees, signed as SPIGA's yaw (positive when the nose points to the
          image right); `cheek.yaw.model`: SPIGA's yaw through
          cheek.calibrate_yaw; `cheek.pitch`: SPIGA's raw pitch;
          `cheek.pitch.ok`: 1 when abs(cheek.pitch) <= 20, the only place pitch
          enters.

        Draws the pose line of the text column in either view whenever it is on.

    Every value is a float; flags are 0.0 or 1.0; NaN means not applicable in
    this view or not determinable: no pupil frame (fewer than 98 landmarks, a
    landmark that isn't finite, or coincident pupils) blanks every frontal and
    oblique output and `cheek.yaw`, the one rule the cheek landmarks follow
    (cheek.locate), so an apex is never reported without the yaw that gates it,
    except `cheek.far`, which follows the face's far_left on every quarter face;
    an outline band too short to search (under three rows between the canthus
    and stomion levels) blanks the rest of the oblique outputs alone; the head
    pose's own values (`cheek.yaw.model`, `cheek.pitch`, `cheek.pitch.ok`)
    always come back. That NaN, never an exception, is this measure's own
    contract: the other measures make no such promise, so analyze() with all of
    them still raises on landmarks replaced by fewer than 98 or non-finite ones
    (Position indexes the pupils and the drawing rounds every point), as it did
    before this measure. A turned head's constructions are still reported, with
    their flag 0, so a chart shows where the head turned rather than a gap.
    "Right" and "left" are the subject's, and the far (measured) cheek of a
    quarter face is the face's far_left, which names the reported side and sets
    `cheek.far`; the near side's oblique outputs are always NaN, so the two
    mirrored photos of a visit line up by column. The yaw every gate uses is
    `cheek.yaw`, measured from the landmarks, and each gate also needs it within
    YAW_AGREE_DEG of the model's calibrated yaw.

    Labels are written at the standard measure text size outside their point,
    away from the face, or on its other side when the outside would cross the
    image's edge (_label()). The text column, written through
    face.analyze_next_pt(), holds at most three headline lines for any one
    face, never one line per output, of five kinds in all, each written by its
    item: on a frontal face "zy.w=130.6 mm (R 65.5 / L 65.1, ratio 0.99)"
    (" (flagged)" appended when zy.ok is 0, "Zygion width") and "cheek ref depth
    R 28.8 / L 28.8 mm" ("Reference points"); on a quarter face "apex.l h=0.79
    dev=0.012 (prior)" ("Malar apex") and "sub.l depth=0.006 (hollow)" or
    "(absent)" ("Submalar point", nothing when its depth is NaN); and the pose
    line, "cheek yaw 44.6 (model 42.3) pitch 4.7"
    ("Head turn"), in either view, with " (flagged)" when the view's ok flag
    (zy.ok on a frontal face, apex.ok on a quarter face) or cheek.pitch.ok is
    0, so a port writes the same five kinds. The submalar line also ends in
    " (flagged)" when sub.ok is 0.

    On a quarter face whose result is flagged, a line for each reason follows,
    starting with FLAG_PREFIX ("flagged: "), so the user sees why rather than
    only that: under "Malar apex" or "Head turn", each of the apex's failed
    gates in the order they are checked ("turned too far (50.1 degrees, needs
    40 to 50)" or "not turned enough (...)", "the turn estimates differ (50.1
    and 43.2 degrees)", "the nose hides the far cheek", "the apex is level with
    the nose tip", "the cheek's edge is unclear (hair or background)"); under
    "Submalar point", "no submalar depth (the edge is unclear below the apex)"
    when its depth is NaN, or "(the nose covers the cheek below the apex)" when
    nose rows are why; under "Head turn", "head tipped up or down (pitch
    25.0, needs within 20)"; and "the far cheek could not be measured" when
    there are no observed points. A frontal face writes no reason lines.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.items = [
            MeasureItem(name, outputs=outputs, views=views)
            for name, views, outputs in ITEMS
        ]
        self.is_frontal, self.is_quarter, self.is_lateral = True, True, False
        self.sync_items()

    def abbrev(self) -> str:
        """
        Return the abbreviation for the cheek measures.

        Returns:
            str: Abbreviation string.
        """
        return "Cheek Measures"

    def calc(self, face: Any, render: bool = True) -> Dict[str, Any]:
        """
        Calculate the cheek measures, drawing what each enabled item draws when
        render is True.

        On a quarter face the observed points come from face.oblique_cheek() when
        the face has that accessor (the result of its last
        find_cheek_landmarks()), and on a frontal face the constructions from
        face.frontal_cheek(); a face without the accessor gets cheek.locate() on
        its landmarks, view and, on a quarter face, its cheek_outline_x, _y and
        _src (None for the contour fallback), the rule find_cheek_landmarks()
        locates by. Same inputs, same values. The reported side is the face's
        far_left; observed points that say otherwise are reported on the face's
        side, with a warning.

        Args:
            face (Any): The face: its landmarks, view (lateral, quarter, far_left),
                head pose (yaw, pitch), pix2mm and, on a quarter face, its outline.
            render (bool): Whether to draw; the values are the same either way.

        Returns:
            Dict[str, Any]: Every declared output (see the class), NaN where not
                applicable in the view or not determinable, or an empty dict for a
                lateral face.
        """
        if face.lateral:
            return {}
        data: Dict[str, Any] = {name: math.nan for name in self.outputs()}
        landmarks = face.landmarks
        frame = cheek.pupil_frame(landmarks)
        yaw = cheek.landmark_yaw(landmarks, frame) if frame is not None else math.nan
        yaw_model = cheek.calibrate_yaw(float(getattr(face, "yaw", math.nan)))
        pitch = float(getattr(face, "pitch", math.nan))
        data["cheek.yaw"] = float(yaw)
        data["cheek.yaw.model"] = float(yaw_model)
        data["cheek.pitch"] = pitch
        # Reported, and folded into no other flag: pitch is uncalibrated.
        data["cheek.pitch.ok"] = _flag(abs(pitch) <= cheek.PITCH_GROSS_RAW)
        # False through NaN, so an unknown yaw fails every yaw gate.
        agree = abs(yaw - yaw_model) <= cheek.YAW_AGREE_DEG
        column: List[str] = []
        reasons: List[str] = []
        if getattr(face, "quarter", False):
            far_left = bool(getattr(face, "far_left", False))
            if hasattr(face, "oblique_cheek"):
                ob = face.oblique_cheek()
            else:
                _, ob = cheek.locate(
                    landmarks,
                    True,
                    False,
                    far_left,
                    getattr(face, "cheek_outline_x", None),
                    getattr(face, "cheek_outline_y", None),
                    getattr(face, "cheek_outline_src", None),
                )
            data["cheek.far"] = _flag(far_left)
            if ob is not None:
                self._fill_oblique(data, ob, far_left, yaw, agree)
            if render and ob is not None:
                outside_left = not far_left
                if self.is_enabled(ITEM_OUTLINE):
                    self._draw_outline(face, ob, far_left)
                if self.is_enabled(ITEM_APEX):
                    self._draw_apex(face, ob, outside_left, data)
                    column.append(self._column_apex(ob, far_left))
                if self.is_enabled(ITEM_SUBMALAR):
                    self._draw_submalar(face, ob, outside_left)
                    column += self._column_submalar(ob, far_left, data)
            if render:
                reasons = self._quarter_reasons(ob, data, yaw, yaw_model, agree)
            view_ok = data["apex.ok"]
        else:
            if hasattr(face, "frontal_cheek"):
                fc = face.frontal_cheek()
            else:
                fc, _ = cheek.locate(landmarks, False, False, False, None, None, None)
            if fc is not None:
                self._fill_frontal(data, fc, float(face.pix2mm), yaw, agree)
            if render and fc is not None:
                if self.is_enabled(ITEM_ZYGION):
                    self._draw_zygion(face, fc, data)
                    column.append(self._column_zygion(data))
                if self.is_enabled(ITEM_REFERENCE):
                    self._draw_reference(face, fc, data)
                    column.append(self._column_reference(data))
            view_ok = data["zy.ok"]
        if render and self.is_enabled(ITEM_HEAD_TURN):
            column.append(self._column_yaw(data, view_ok))
        column += [FLAG_PREFIX + reason for reason in reasons]
        for txt in column:
            face.write_text(face.analyze_next_pt(txt), txt)
        return select_outputs(data, self.items)

    @staticmethod
    def _fill_frontal(
        data: Dict[str, Any],
        fc: cheek.FrontalCheek,
        pix2mm: float,
        yaw: float,
        agree: bool,
    ) -> None:
        """
        The frontal items' values and gates from the constructions (their
        coordinates are the Landmarks measure's).

        Args:
            data (Dict[str, Any]): The outputs, filled in place.
            fc (cheek.FrontalCheek): The six constructions, px.
            pix2mm (float): The face's scale, mm per px.
            yaw (float): cheek.yaw, degrees.
            agree (bool): Whether cheek.yaw and cheek.yaw.model agree.
        """
        zy_r = float(fc.zygion_r_px * pix2mm)
        zy_l = float(fc.zygion_l_px * pix2mm)
        data["zy.r"], data["zy.l"] = zy_r, zy_l
        data["zy.w"] = zy_r + zy_l
        data["zy.ratio"] = float(util.symmetry_ratio(zy_r, zy_l))
        data["zy.r.clip"] = float(fc.zygion_r_clip)
        data["zy.l.clip"] = float(fc.zygion_l_clip)
        pose_ok = abs(yaw) <= cheek.FRONTAL_YAW_GATE and agree
        data["zy.r.ok"] = _flag(pose_ok and fc.zygion_r_clip == cheek.CLIP_NONE)
        data["zy.l.ok"] = _flag(pose_ok and fc.zygion_l_clip == cheek.CLIP_NONE)
        data["zy.ok"] = _flag(data["zy.r.ok"] == 1.0 and data["zy.l.ok"] == 1.0)
        data["zy.ratio.ok"] = _flag(
            data["zy.ok"] == 1.0 and abs(yaw) <= cheek.ZYGION_RATIO_YAW_GATE
        )
        data["cr.r.d"] = float(fc.cheek_ref_r_px * pix2mm)
        data["cr.l.d"] = float(fc.cheek_ref_l_px * pix2mm)
        data["cr.ok"] = data["sr.ok"] = _flag(pose_ok)

    @staticmethod
    def _fill_oblique(
        data: Dict[str, Any],
        ob: cheek_outline.ObliqueCheek,
        far_left: bool,
        yaw: float,
        agree: bool,
    ) -> None:
        """
        The quarter items' values and gates from the observed points (their
        coordinates are the Landmarks measure's), for the far side only: the
        face's, which also set cheek.far. Observed points that name the other
        side (never from find_cheek_landmarks(), which observes with the face's
        far side) are reported on the face's, with a warning, so cheek.far and the
        reported side never disagree.

        Args:
            data (Dict[str, Any]): The outputs, filled in place.
            ob (cheek_outline.ObliqueCheek): The observed points and their flags.
            far_left (bool): The face's far side: whether the far cheek is the
                subject's left.
            yaw (float): cheek.yaw, degrees.
            agree (bool): Whether cheek.yaw and cheek.yaw.model agree.
        """
        if bool(ob.far_left) != far_left:
            logger.warning(
                "The observed cheek points say the far cheek is the subject's %s, "
                "the face says the %s: reported on the face's side.",
                "left" if ob.far_left else "right",
                "left" if far_left else "right",
            )
        s = "l" if far_left else "r"
        data[f"apex.{s}.h"] = float(ob.apex_h)
        data[f"apex.{s}.dev"] = float(ob.apex_dev)
        data[f"apex.{s}.prior"] = _flag(ob.apex_prior)
        data[f"apex.{s}.near_nose"] = _flag(ob.apex_near_nose)
        data[f"sub.{s}.h"] = float(ob.submalar_h)
        data[f"sub.{s}.depth"] = float(ob.submalar_depth)
        data[f"sub.{s}.depth.px"] = float(ob.submalar_depth_px)
        data[f"sub.{s}.present"] = float(ob.submalar_present)
        data["cheek.mask"] = float(ob.mask_fraction)
        data["cheek.nose"] = _flag(ob.nose_on)
        data["cheek.nose.gap"] = float(ob.nose_gap)
        data["cheek.scale"] = float(ob.scale)
        data["apex.ok"] = _flag(
            abs(abs(yaw) - cheek.OBLIQUE_YAW) <= cheek.OBLIQUE_YAW_GATE
            and agree
            and not ob.nose_on
            and not ob.apex_near_nose
            and ob.window_mask >= cheek_outline.OUTLINE_MIN_MASK
        )
        data["sub.ok"] = _flag(
            data["apex.ok"] == 1.0 and not math.isnan(ob.submalar_depth)
        )

    @staticmethod
    def _label(
        face: Any,
        point: ArrayLike,
        text: str,
        color: Tuple[int, int, int],
        outside_left: bool,
    ) -> None:
        """
        Write a point's label outside it at the standard measure text size:
        LABEL_DY px above the point, and LABEL_DX px to its left (the text's
        right end there) or to its right.

        A label that would cross the image's edge on its outside (start before
        x = 0 or end past face.width) is written on the point's other side
        instead; if it crosses on both sides, its x is clamped so the text stays
        inside the image.

        Args:
            face (Any): The face to draw on.
            point (ArrayLike): The labeled point, (x, y) px.
            text (str): The label.
            color (Tuple[int, int, int]): Its color, RGB.
            outside_left (bool): Whether the label goes to the point's left (the
                subject's right side, whose outside is the image left).
        """
        x, y = _pt(point)
        width = int(face.calc_text_size(text)[0][0])
        left, right = x - width - LABEL_DX, x + LABEL_DX
        candidates = (left, right) if outside_left else (right, left)
        for text_x in candidates:
            if text_x >= 0 and text_x + width <= face.width:
                break
        else:
            text_x = max(0, min(candidates[0], face.width - width))
        face.write_text((text_x, y - LABEL_DY), text, color=color)

    def _draw_zygion(
        self, face: Any, fc: cheek.FrontalCheek, data: Dict[str, Any]
    ) -> None:
        """
        The zygion level line and the two zygion dots with their labels.

        Args:
            face (Any): The face to draw on.
            fc (cheek.FrontalCheek): The constructions.
            data (Dict[str, Any]): The outputs, for the per-side ok flags.
        """
        frame = fc.frame
        overhang = LEVEL_OVERHANG_IPD * frame.ipd
        level_r = frame.point(-fc.zygion_r_px - overhang, fc.zygion_level)
        level_l = frame.point(fc.zygion_l_px + overhang, fc.zygion_level)
        if _finite(level_r, level_l):
            face.line(_pt(level_r), _pt(level_l), ORANGE, 2)
        for point, clip, ok, outside_left in (
            (fc.zygion_r, fc.zygion_r_clip, data["zy.r.ok"], True),
            (fc.zygion_l, fc.zygion_l_clip, data["zy.l.ok"], False),
        ):
            if not _finite(point):
                continue
            face.circle(_pt(point), WHITE, DOT_RIM)
            face.circle(_pt(point), BLUE if ok == 1.0 else GRAY, DOT_RADIUS)
            label = "zygion" + (" (clipped)" if clip != cheek.CLIP_NONE else "")
            self._label(face, point, label, BLUE, outside_left)

    def _draw_reference(
        self, face: Any, fc: cheek.FrontalCheek, data: Dict[str, Any]
    ) -> None:
        """
        The canthus verticals and the open rings of the reference points with
        their labels.

        Args:
            face (Any): The face to draw on.
            fc (cheek.FrontalCheek): The constructions.
            data (Dict[str, Any]): The outputs, for cr.ok.
        """
        color = ORANGE if data["cr.ok"] == 1.0 else GRAY
        for canthus, foot in ((fc.canthus_r, fc.foot_r), (fc.canthus_l, fc.foot_l)):
            if _finite(canthus, foot):
                face.line(_pt(canthus), _pt(foot), ORANGE, 3)
        for point, text, outside_left in (
            (fc.cheek_ref_r, "cheek reference point", True),
            (fc.submalar_ref_r, "submalar reference point", True),
            (fc.cheek_ref_l, "cheek reference point", False),
            (fc.submalar_ref_l, "submalar reference point", False),
        ):
            if not _finite(point):
                continue
            face.circle(_pt(point), color, DOT_RADIUS, thickness=RING_THICKNESS)
            self._label(face, point, text, ORANGE, outside_left)

    @staticmethod
    def _draw_outline(
        face: Any, ob: cheek_outline.ObliqueCheek, far_left: bool
    ) -> None:
        """
        The far outline colored by source, the three level ticks and the chord.

        Args:
            face (Any): The face to draw on.
            ob (cheek_outline.ObliqueCheek): The observed points.
            far_left (bool): The face's far side, which decides which way is
                inward.
        """
        points, src = ob.outline, ob.outline_src
        n = len(points)
        # Each run of rows with one source is one polyline, through the first
        # row of the next run too, so the segment between two rows takes the
        # upper row's color and no segment is skipped.
        start = 0
        while start < n - 1:
            end = start
            while end < n - 1 and src[end] == src[start]:
                end += 1
            face.polyline(
                [_pt(p) for p in points[start : end + 1]],
                OUTLINE_COLORS.get(int(src[start]), GRAY),
                OUTLINE_THICKNESS,
            )
            start = end
        a, b = ob.chord
        out = 1.0 if far_left else -1.0
        length = OBLIQUE_LEVEL_LENGTH * ob.scale
        for name, tick_start in (
            ("canthus", a),
            ("stomion", b),
            ("chin", ob.levels["chin"][0]),
        ):
            direction = ob.levels[name][1]
            inward = direction if direction[0] * out < 0.0 else -direction
            face.line(_pt(tick_start), _pt(tick_start + inward * length), ORANGE, 1)
        face.line(_pt(a), _pt(b), ORANGE, 2)

    def _draw_apex(
        self,
        face: Any,
        ob: cheek_outline.ObliqueCheek,
        outside_left: bool,
        data: Dict[str, Any],
    ) -> None:
        """
        The malar apex's dot and label.

        Args:
            face (Any): The face to draw on.
            ob (cheek_outline.ObliqueCheek): The observed points.
            outside_left (bool): Whether the label goes to the point's left (the
                near side of a far-right cheek: not far_left).
            data (Dict[str, Any]): The outputs, for apex.ok.
        """
        face.circle(_pt(ob.apex), WHITE, DOT_RIM)
        face.circle(_pt(ob.apex), RED if data["apex.ok"] == 1.0 else GRAY, DOT_RADIUS)
        self._label(face, ob.apex, "malar apex" + _apex_note(ob), RED, outside_left)

    def _draw_submalar(
        self, face: Any, ob: cheek_outline.ObliqueCheek, outside_left: bool
    ) -> None:
        """
        The submalar point's dot and label, when a hollow is present.

        Args:
            face (Any): The face to draw on.
            ob (cheek_outline.ObliqueCheek): The observed points.
            outside_left (bool): Whether the label goes to the point's left.
        """
        # Only a present hollow is drawn: an absent point is wherever the
        # outline comes nearest the chord, on a convex cheek a row beside one of
        # the chord's ends (73 of 323 absent points on studio photos sat on the
        # apex dot, the rest at the stomion end), so a ring there marked
        # nothing and overprinted the apex label.
        if ob.submalar_present == 1.0:
            face.circle(_pt(ob.submalar), GREEN, DOT_RADIUS)
            self._label(face, ob.submalar, "submalar point", GREEN, outside_left)

    def _quarter_reasons(
        self,
        ob: Optional[cheek_outline.ObliqueCheek],
        data: Dict[str, Any],
        yaw: float,
        yaw_model: float,
        agree: bool,
    ) -> List[str]:
        """
        Why a quarter face's result is flagged, for the enabled items' lines:
        the far cheek not measured at all (no observed points), under "Malar
        apex", "Submalar point" or "Head turn"; the apex's failed gates
        (_apex_reasons()) when apex.ok is 0, under "Malar apex" or "Head turn",
        whose lines carry that flag; a submalar depth that couldn't be read,
        under "Submalar point"; and a gross pitch, under "Head turn".

        Args:
            ob (Optional[cheek_outline.ObliqueCheek]): The observed points, or
                None when the far cheek couldn't be measured.
            data (Dict[str, Any]): The outputs.
            yaw (float): cheek.yaw, degrees.
            yaw_model (float): cheek.yaw.model, degrees.
            agree (bool): Whether the two yaws agree.

        Returns:
            List[str]: The phrases, in that order; empty when nothing is flagged
                or no item that would say so is on.
        """
        apex, submalar = self.is_enabled(ITEM_APEX), self.is_enabled(ITEM_SUBMALAR)
        head_turn = self.is_enabled(ITEM_HEAD_TURN)
        reasons: List[str] = []
        if ob is None:
            if apex or submalar or head_turn:
                reasons.append("the far cheek could not be measured")
        else:
            if data["apex.ok"] != 1.0 and (apex or head_turn):
                reasons += _apex_reasons(ob, yaw, yaw_model, agree)
            if submalar and math.isnan(ob.submalar_depth):
                reasons.append(_submalar_reason(ob))
        if head_turn and data["cheek.pitch.ok"] != 1.0:
            pitch = _beyond(data["cheek.pitch"], cheek.PITCH_GROSS_RAW)
            reasons.append(
                f"head tipped up or down (pitch {pitch}, needs within"
                f" {cheek.PITCH_GROSS_RAW:.0f})"
            )
        return reasons

    @staticmethod
    def _column_zygion(data: Dict[str, Any]) -> str:
        """The zygion headline: the width, the two midline distances, the ratio."""
        txt = (
            f"zy.w={data['zy.w']:.1f} mm (R {data['zy.r']:.1f} / L {data['zy.l']:.1f},"
            f" ratio {data['zy.ratio']:.2f})"
        )
        return txt + ("" if data["zy.ok"] == 1.0 else " (flagged)")

    @staticmethod
    def _column_reference(data: Dict[str, Any]) -> str:
        """The reference headline: the cheek reference points' depths."""
        return f"cheek ref depth R {data['cr.r.d']:.1f} / L {data['cr.l.d']:.1f} mm"

    @staticmethod
    def _column_apex(ob: cheek_outline.ObliqueCheek, far_left: bool) -> str:
        """The apex headline, named for the face's far side."""
        s = "l" if far_left else "r"
        return f"apex.{s} h={ob.apex_h:.2f} dev={ob.apex_dev:.3f}{_apex_note(ob)}"

    @staticmethod
    def _column_submalar(
        ob: cheek_outline.ObliqueCheek, far_left: bool, data: Dict[str, Any]
    ) -> List[str]:
        """The submalar headline when its depth is known (nothing otherwise),
        named for the face's far side, flagged when sub.ok is 0."""
        if math.isnan(ob.submalar_depth):
            return []
        s = "l" if far_left else "r"
        state = "hollow" if ob.submalar_present == 1.0 else "absent"
        # Rounded first and + 0.0, so a depth just below zero reads 0.000, not
        # -0.000.
        depth = round(ob.submalar_depth, 3) + 0.0
        flagged = "" if data["sub.ok"] == 1.0 else " (flagged)"
        return [f"sub.{s} depth={depth:.3f} ({state}){flagged}"]

    @staticmethod
    def _column_yaw(data: Dict[str, Any], view_ok: float) -> str:
        """The pose headline, flagged when the view's gate or the pitch failed."""
        txt = (
            f"cheek yaw {data['cheek.yaw']:.1f} (model {data['cheek.yaw.model']:.1f})"
            f" pitch {data['cheek.pitch']:.1f}"
        )
        flagged = view_ok != 1.0 or data["cheek.pitch.ok"] != 1.0
        return txt + (" (flagged)" if flagged else "")
