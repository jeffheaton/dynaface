from PyQt6.QtGui import QUndoCommand


class CommandClip(QUndoCommand):
    def __init__(self, target: object, new_begin: int, new_end: int):
        super().__init__()
        self._target = target
        self._new_begin = new_begin
        self._new_end = new_end
        self._old_begin = target._frame_begin
        self._old_end = target._frame_end

    def redo(self):
        self._target.set_video_range(self._new_begin, self._new_end)

    def undo(self):
        self._target.set_video_range(self._old_begin, self._old_end)
