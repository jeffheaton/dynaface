import gzip
import pickle

from jth_ui import utl_classes
import dynaface

DOC_HEADER = "header"
DOC_HEADER_VERSION = "version"
DOC_HEADER_FPS = "fps"
DOC_BODY = "body"
DOC_BODY_FACE = "face"
DOC_BODY_MEASURES = "measures"
DOC_BODY_MEASURE = "measure"
DOC_BODY_MEASURE_ITEMS = "items"
DOC_BODY_FRAMES = "frames"

DOC_NAME = "name"
DOC_ENABLED = "enabled"


class DynafaceDocument:
    def __init__(self):
        self._version = 1
        self.face = None
        self.frames = []
        self.measures = []
        self.fps = 0

    def save(self, filename: str):
        measures = self._save_measures(self.measures)

        doc = {
            DOC_HEADER: {
                DOC_HEADER_VERSION: self._version,
                DOC_HEADER_FPS: self.fps,
            },
            DOC_BODY: {DOC_BODY_MEASURES: measures, DOC_BODY_FRAMES: self.frames},
        }

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
