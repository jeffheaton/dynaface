DOC_TYPE_VIDEO = "video"
DOC_TYPE_IMAGE = "image"


class DynafaceDocument:
    def __init__(self, type):
        self._version = 1
        self._type = type
