import gzip
import pickle

from dynaface.const import Pose
from jth_ui import utl_classes
from utl_general import assert_standard_python

import dynaface

DOC_HEADER = "header"
DOC_HEADER_VERSION = "version"
DOC_HEADER_FPS = "fps"
DOC_HEADER_VIEW = "view"
DOC_HEADER_LATERAL = "lateral"
DOC_HEADER_POSE = "pose"
DOC_BODY = "body"
DOC_BODY_FACE = "face"
DOC_BODY_MEASURES = "measures"
DOC_BODY_MEASURE = "measure"
DOC_BODY_MEASURE_ITEMS = "items"
DOC_BODY_FRAMES = "frames"
DOC_BODY_LATERAL_LANDMARKS = "lateral_landmarks"
DOC_BODY_SAGITTAL_X = "sagittal_x"
DOC_BODY_SAGITTAL_Y = "sagittal_y"

DOC_NAME = "name"
DOC_ENABLED = "enabled"


class DynafaceDocument:
    def __init__(self):
        self._version = 1
        self.face = None
        self.frames = []
        self.measures = []
        self.fps = 0
        self.lateral = False
        self.lateral_landmarks = None
        self.sagittal_x = None
        self.sagittal_y = None
        self.pose = None

    def save(self, filename: str):
        measures = self._save_measures(self.measures)

        doc = {
            DOC_HEADER: {
                DOC_HEADER_VERSION: self._version,
                DOC_HEADER_FPS: self.fps,
                DOC_HEADER_LATERAL: self.lateral,
                DOC_HEADER_POSE: self.pose.value,
            },
            DOC_BODY: {
                DOC_BODY_MEASURES: measures,
                DOC_BODY_FRAMES: self.frames,
                DOC_BODY_LATERAL_LANDMARKS: self.lateral_landmarks,
                DOC_BODY_SAGITTAL_X: self.sagittal_x,
                DOC_BODY_SAGITTAL_Y: self.sagittal_y,
            },
        }
        assert_standard_python(doc)

        with gzip.open(filename, "wb") as f:
            pickle.dump(doc, f)

    def load(self, filename: str):
        try:
            with gzip.open(filename, "rb") as f:
                doc = pickle.load(f)
        except gzip.BadGzipFile:
            raise TypeError(
                f"The file '{filename}' does not appear to be a valid Dynaface document."
            )

        # load
        measures = self._load_measures(doc[DOC_BODY][DOC_BODY_MEASURES])
        self._add_missing_measures(measures, dynaface.measures.all_measures())

        self.fps = doc[DOC_HEADER][DOC_HEADER_FPS]
        self.frames = doc[DOC_BODY][DOC_BODY_FRAMES]
        self.measures = measures
        if DOC_HEADER_LATERAL not in doc[DOC_HEADER]:
            # If the lateral key is not present, set it to False
            self.lateral = False
            self.lateral_landmarks = None
            self.sagittal_x = None
            self.sagittal_y = None
        else:
            # If the key is present, set it to the value in the document
            # This allows for backward compatibility with older documents
            # that do not have this key.
            self.lateral = doc[DOC_HEADER][DOC_HEADER_LATERAL]
            self.lateral_landmarks = doc[DOC_BODY][DOC_BODY_LATERAL_LANDMARKS]
            self.sagittal_x = doc[DOC_BODY][DOC_BODY_SAGITTAL_X]
            self.sagittal_y = doc[DOC_BODY][DOC_BODY_SAGITTAL_Y]
            self.pose = (
                Pose(doc[DOC_HEADER][DOC_HEADER_POSE])
                if DOC_HEADER_POSE in doc[DOC_HEADER]
                else Pose.FRONTAL
            )

    def _load_measures(self, measures):
        result = []
        for measure in measures:
            name = measure[DOC_NAME]
            enabled = measure[DOC_ENABLED]
            source_items = measure[DOC_BODY_MEASURE_ITEMS]
            obj = utl_classes.create_instance_from_full_name(name)
            # Make sure we have a class to handle this measure
            if obj:
                obj.enabled = enabled
                self._sync_items(source_items, obj)
                result.append(obj)
        return result

    def _sync_items(self, source_items, obj):
        """Update the disabled flag on the target items, based on the source"""

        for item in source_items:
            obj.set_item_enabled(item[DOC_NAME], item[DOC_ENABLED])

    def _save_measures(self, measures):
        result = []
        for measure in measures:
            items = []
            for item in measure.items:
                items.append({DOC_NAME: item.name, DOC_ENABLED: item.enabled})

            name = utl_classes.get_class_full_name(measure)
            measure_encoded = {
                DOC_NAME: name,
                DOC_ENABLED: measure.enabled,
                DOC_BODY_MEASURE_ITEMS: items,
            }
            result.append(measure_encoded)

        return result

    def _has_measure(self, doc_measures, measure):
        for m in doc_measures:
            if m.abbrev() == measure.abbrev():
                return True
        return False

    def _add_missing_measures(self, doc_measures, all_measures):

        for measure in all_measures:
            if not self._has_measure(doc_measures, measure):
                doc_measures.append(measure)
                measure.set_enabled(False)
