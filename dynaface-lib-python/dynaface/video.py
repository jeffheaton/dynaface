"""
Measure a video frame by frame: the values behind its chart, a CSV file of them,
the frame where an output peaks, and the evaluation the Dynaface apps report.

Build a FrameSeries while analyzing the frames, adding each frame's analyze()
results, or from saved states (AnalyzeFace.dump_state()) with
FrameSeries.from_states().
"""

import csv
import math
import os
from dataclasses import dataclass
from typing import IO, Any, Dict, Iterable, List, Optional, Sequence, Union

from dynaface import measures as dynaface_measures
from dynaface.facial import AnalyzeFace
from dynaface.measures_base import MeasureBase


def _number(value: Any) -> float:
    """An output's value as a float: NaN for one that's missing or not a number."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _ratio(a: float, b: float) -> float:
    """a over b, or NaN if b is 0."""
    return a / b if b else math.nan


def _csv_value(value: float) -> str:
    """A value for the CSV: at most 4 decimal places, without trailing zeros, and
    empty when it isn't a finite number."""
    if not math.isfinite(value):
        return ""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


@dataclass
class Moment:
    """
    A frame that the evaluation picks out, with the outputs it compares there.

    Attributes:
        frame (int): The frame's number in the series.
        time (float): Its time in seconds (frame / fps), or NaN if the series has
            no frame rate.
        dental_area (float): The dental display, in square millimeters.
        eye_left (float): The left eye's area, in square millimeters.
        eye_right (float): The right eye's area, in square millimeters.
    """

    frame: int
    time: float
    dental_area: float
    eye_left: float
    eye_right: float

    @property
    def eye_total(self) -> float:
        """Both eyes' areas added together."""
        return self.eye_left + self.eye_right

    @property
    def eye_left_over_right(self) -> float:
        """The left eye's area over the right's, or NaN if the right's is 0."""
        return _ratio(self.eye_left, self.eye_right)

    @property
    def eye_right_over_left(self) -> float:
        """The right eye's area over the left's, or NaN if the left's is 0."""
        return _ratio(self.eye_right, self.eye_left)


@dataclass
class Evaluation:
    """
    The eyes at the widest smile and the mouth at the widest eye opening: the
    comparison used in studies of oral-ocular synkinesis, which the Dynaface apps
    show as their Evaluation report.

    Attributes:
        widest_smile (Moment): The frame with the largest dental_area.
        widest_eyes (Moment): The frame with the largest eye.left + eye.right.
    """

    widest_smile: Moment
    widest_eyes: Moment


class FrameSeries:
    """
    Measure outputs frame by frame: one row per frame, one column per output.

    Attributes:
        names (List[str]): The outputs, in column order: the enabled items of the
            enabled measures (see dynaface.measures.get_all_items()).
        frames (List[int]): Each row's frame number.
        values (Dict[str, List[float]]): Each output's value in every row. A value
            is NaN where the frame had no face or it couldn't be determined.
        fps (float): Frames per second, for the time of each frame; 0 if unknown.
    """

    def __init__(self, measures: Iterable[MeasureBase], fps: float = 0.0) -> None:
        """
        Start an empty series for the outputs of these measures.

        Args:
            measures (Iterable[MeasureBase]): The measures whose outputs to keep,
                the same ones the frames are analyzed with.
            fps (float): Frames per second, for the time of each frame; 0 if
                unknown.
        """
        self.names: List[str] = list(
            dict.fromkeys(dynaface_measures.get_all_items(list(measures)))
        )
        self.frames: List[int] = []
        self.values: Dict[str, List[float]] = {name: [] for name in self.names}
        self.fps: float = float(fps)

    @classmethod
    def from_states(
        cls,
        states: Sequence[List[Any]],
        measures: Optional[List[MeasureBase]] = None,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
        fps: float = 0.0,
    ) -> "FrameSeries":
        """
        Measure saved states again, frame by frame, without drawing.

        Each row is numbered by its state's position in states, so the frames of a
        trimmed range keep the numbers they have in the whole video.

        Args:
            states (Sequence[List[Any]]): States from AnalyzeFace.dump_state().
            measures (Optional[List[MeasureBase]]): The measures to run; all of
                them (dynaface.measures.all_measures()) if None.
            start (int): The first state to measure.
            stop (Optional[int]): The state to stop before; the end if None.
            step (int): Measure every step-th state, such as every 10th for a
                quick chart of a long video.
            fps (float): Frames per second, for the time of each frame.

        Returns:
            FrameSeries: One row per measured state.

        Raises:
            ValueError: If step is less than 1.
        """
        if step < 1:
            raise ValueError("The step must be at least 1.")
        face = AnalyzeFace(measures)
        series = cls(face.measures, fps)
        end = len(states) if stop is None else stop
        for frame in range(start, end, step):
            face.load_state(states[frame])
            series.add(frame, face.analyze(render=False))
        return series

    def __len__(self) -> int:
        return len(self.frames)

    def add(self, frame: int, results: Optional[Dict[str, Any]]) -> None:
        """
        Add a row: one frame's analyze() results.

        Args:
            frame (int): The frame's number, such as its position in the video.
            results (Optional[Dict[str, Any]]): What analyze() returned for it;
                None for a frame without a face, whose values are all NaN.
        """
        self.frames.append(int(frame))
        for name in self.names:
            value = None if results is None else results.get(name)
            self.values[name].append(_number(value))

    def time(self, frame: int) -> float:
        """A frame's time in seconds, or NaN if the series has no frame rate."""
        return frame / self.fps if self.fps > 0 else math.nan

    def rows(self) -> List[Dict[str, float]]:
        """
        The series as one dict per row, with its frame, time and every output, for
        example to load into pandas with pandas.DataFrame(series.rows()).
        """
        return [
            {
                "frame": frame,
                "time": self.time(frame),
                **{name: self.values[name][row] for name in self.names},
            }
            for row, frame in enumerate(self.frames)
        ]

    def peak(self, *names: str) -> Optional[int]:
        """
        The frame where these outputs, added together, are largest. The widest
        smile is peak("dental_area"), and the widest eye opening is
        peak("eye.left", "eye.right"). Rows where any of them is NaN are skipped,
        and of equal rows, the first wins.

        Args:
            names (str): One or more outputs in the series.

        Returns:
            Optional[int]: The frame's number, or None if no row has all of them.

        Raises:
            KeyError: If an output isn't in the series.
        """
        row = self._peak_row(names)
        return None if row is None else self.frames[row]

    def evaluate(self) -> Optional[Evaluation]:
        """
        Find the widest smile and the widest eye opening, and the eyes and mouth
        at each, from the dental_area, eye.left and eye.right outputs. Build the
        series with AnalyzeDentalArea and AnalyzeEyeArea, with those items
        enabled, to have them.

        Returns:
            Optional[Evaluation]: The two moments, or None if the series lacks one
                of those outputs or no frame has a value for it.
        """
        if any(n not in self.values for n in ("dental_area", "eye.left", "eye.right")):
            return None
        smile = self._peak_row(("dental_area",))
        eyes = self._peak_row(("eye.left", "eye.right"))
        if smile is None or eyes is None:
            return None
        return Evaluation(self._moment(smile), self._moment(eyes))

    def write_csv(self, file: Union[str, "os.PathLike[str]", IO[str]]) -> None:
        """
        Write the series as CSV: a header of frame, time and the outputs, then one
        row per frame. The time is in seconds, to 2 decimal places, and is left
        empty without a frame rate. Values have at most 4 decimal places, and a
        NaN value is left empty.

        Args:
            file (Union[str, os.PathLike, IO[str]]): A path, or an open text file
                (opened with newline="").
        """
        if isinstance(file, (str, os.PathLike)):
            with open(file, "w", newline="", encoding="utf-8") as f:
                self._write_csv(f)
        else:
            self._write_csv(file)

    def _write_csv(self, f: IO[str]) -> None:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["frame", "time", *self.names])
        for row, frame in enumerate(self.frames):
            time = self.time(frame)
            values = (self.values[name][row] for name in self.names)
            writer.writerow(
                [
                    frame,
                    "" if math.isnan(time) else f"{time:.2f}",
                    *(_csv_value(value) for value in values),
                ]
            )

    def _peak_row(self, names: Sequence[str]) -> Optional[int]:
        if not names:
            raise ValueError("Name at least one output.")
        columns = [self.values[name] for name in names]
        best: Optional[int] = None
        best_total = -math.inf
        for row in range(len(self.frames)):
            parts = [column[row] for column in columns]
            if any(math.isnan(part) for part in parts):
                continue
            total = sum(parts)
            if total > best_total:
                best, best_total = row, total
        return best

    def _moment(self, row: int) -> Moment:
        frame = self.frames[row]
        return Moment(
            frame=frame,
            time=self.time(frame),
            dental_area=self.values["dental_area"][row],
            eye_left=self.values["eye.left"][row],
            eye_right=self.values["eye.right"][row],
        )
