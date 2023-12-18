import pickle
from facial_analysis.facial import STATS, AnalyzeFace, load_face_image
from jth_ui import utl_classes

DOC_TYPE_VIDEO = "video"
DOC_TYPE_IMAGE = "image"

DOC_HEADER = "header"
DOC_HEADER_TYPE = "type"
DOC_HEADER_VERSION = "version"
DOC_BODY = "body"
DOC_BODY_FACE = "face"
DOC_BODY_CALC = "calc"
DOC_BODY_CALC_NAME = "name"
DOC_BODY_CALC_ENABLED = "enabled"


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

    def save(self, filename):
        calcs = []
        for calc in self.face.calcs:
            name = utl_classes.get_class_full_name(calc)
            calculation = {
                DOC_BODY_CALC_NAME: name,
                DOC_BODY_CALC_ENABLED: calc.enabled,
            }
            calcs.append(calculation)

        doc = {
            DOC_HEADER: {
                DOC_HEADER_TYPE: self._type,
                DOC_HEADER_VERSION: self._version,
            },
            DOC_BODY: {DOC_BODY_FACE: self.face.dump_state(), DOC_BODY_CALC: calcs},
        }

        with open(filename, "wb") as f:
            pickle.dump(doc, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            doc = pickle.load(f)

        stats = []
        for calc in doc[DOC_BODY][DOC_BODY_CALC]:
            name = calc[DOC_BODY_CALC_NAME]
            enabled = calc[DOC_BODY_CALC_ENABLED]
            cls = utl_classes.create_instance_from_full_name(name)
            cls.enabled = enabled
            stats.append(cls)

        state = doc[DOC_BODY][DOC_BODY_FACE]
        self.face = AnalyzeFace(stats)
        self.face.load_state(state)
