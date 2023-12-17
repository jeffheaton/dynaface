import pickle

DOC_TYPE_VIDEO = "video"
DOC_TYPE_IMAGE = "image"

DOC_HEADER = "header"
DOC_HEADER_TYPE = "type"
DOC_HEADER_VERSION = "version"


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

    def save(self, filename):
        doc = {
            DOC_HEADER: {DOC_HEADER_TYPE: self._type, DOC_HEADER_VERSION: self._version}
        }
        with open(filename, "wb") as f:
            pickle.dump(doc, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            doc = pickle.load(f)
