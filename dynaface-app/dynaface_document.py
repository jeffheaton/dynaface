import pickle
from facial_analysis.facial import AnalyzeFace, load_face_image
from jth_ui import utl_classes

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


def check_dyfc_type(filename) -> None:
    with open(filename, "rb") as f:
        doc = pickle.load(f)

    v = doc[DOC_HEADER][DOC_HEADER_VERSION]

    if v > 1:
        return "Error, this application can only read Dynaface version 1 document files. Please update Dynaface."


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

        with open(filename, "wb") as f:
            pickle.dump(doc, f)

    def load(self, filename: str):
        with open(filename, "rb") as f:
            doc = pickle.load(f)

        # load
        measures = self._load_measures(doc[DOC_BODY][DOC_BODY_MEASURES])

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
