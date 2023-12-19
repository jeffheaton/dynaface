import pickle
from facial_analysis.facial import AnalyzeFace, load_face_image
from jth_ui import utl_classes

DOC_TYPE_VIDEO = "video"
DOC_TYPE_IMAGE = "image"

DOC_HEADER = "header"
DOC_HEADER_TYPE = "type"
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


def check_dyfc_type(filename):
    with open(filename, "rb") as f:
        doc = pickle.load(f)

    v = doc[DOC_HEADER][DOC_HEADER_VERSION]
    t = doc[DOC_HEADER][DOC_HEADER_TYPE]

    if v > 1:
        return "Error, this application can only read Dynaface version 1 document files. Please update Dynaface."

    if t != "video" and t != "image":
        return "Error, this application only supports Dynaface video and image files."

    return t


class DynafaceDocument:
    def __init__(self, type):
        self._version = 1
        self._type = type
        self.face = None
        self.frames = []
        self.measures = []
        self.fps = 0

    def save(self, filename: str):
        if self._type == DOC_TYPE_IMAGE:
            self._save_image(filename)
        elif self._type == DOC_TYPE_VIDEO:
            self._save_video(filename)

    def _save_image(self, filename: str):
        measures = self._save_measures(self.face.measures)

        doc = {
            DOC_HEADER: {
                DOC_HEADER_TYPE: self._type,
                DOC_HEADER_VERSION: self._version,
            },
            DOC_BODY: {
                DOC_BODY_FACE: self.face.dump_state(),
                DOC_BODY_MEASURES: measures,
            },
        }

        with open(filename, "wb") as f:
            pickle.dump(doc, f)

    def _save_video(self, filename: str):
        measures = self._save_measures(self.measures)

        doc = {
            DOC_HEADER: {
                DOC_HEADER_TYPE: self._type,
                DOC_HEADER_VERSION: self._version,
                DOC_HEADER_FPS: self.fps,
            },
            DOC_BODY: {DOC_BODY_MEASURES: measures, DOC_BODY_FRAMES: self.frames},
        }

        with open(filename, "wb") as f:
            pickle.dump(doc, f)

    def load(self, filename: str):
        if self._type == DOC_TYPE_IMAGE:
            self._load_image(filename)
        elif self._type == DOC_TYPE_VIDEO:
            self._load_video(filename)

    def _load_image(self, filename: str):
        with open(filename, "rb") as f:
            doc = pickle.load(f)

        measures = self._load_measures(doc[DOC_BODY][DOC_BODY_MEASURES])

        state = doc[DOC_BODY][DOC_BODY_FACE]
        self.face = AnalyzeFace(measures)
        self.face.load_state(state)

    def _load_video(self, filename: str):
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
