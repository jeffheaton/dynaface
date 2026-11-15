"""
Saving a file so that a save that fails part way leaves the old file as it was.
"""

import contextlib
import ctypes
import errno
import functools
import getpass
import hashlib
import io
import logging
import os
import re
import secrets
import shutil
import socket
import stat
import struct
import sys
import tempfile
import time
from typing import Any, BinaryIO, Callable, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# On Windows a virus scanner, the search indexer or a sync client often holds a
# file it has just seen written for a moment, and renaming or removing the file
# fails meanwhile; such a step is tried again for about a second, as pip does.
_WINDOWS = os.name == "nt"
_ATTEMPTS = 10
_RETRY_DELAY = 0.1

# Flags for os.open() that not every system has. A file opened only to look at
# it is opened without waiting, which a FIFO put in its place would.
_BINARY = getattr(os, "O_BINARY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)

# access(2)'s test, on macOS, for whether a file can be removed (sys/unistd.h).
_DELETE_OK = 1 << 12

# Windows file attributes a new file never has, and ones it has only when its
# folder gives them to new files.
_NEVER_NEW = stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM
_FROM_FOLDER = stat.FILE_ATTRIBUTE_COMPRESSED | stat.FILE_ATTRIBUTE_ENCRYPTED

# copyfile(3)'s flag for a file's extended attributes (copyfile.h), the type of
# ACL macOS files have (sys/acl.h), what reading one fails with for a file that
# has none or is on a volume that keeps none (FAT, exFAT), and what setting one
# fails with on such a volume.
_COPYFILE_XATTR = 1 << 2
_ACL_TYPE_EXTENDED = 0x00000100
_NO_ACL = {errno.ENOENT, errno.ENOTSUP, errno.EOPNOTSUPP}
_NO_ACLS_KEPT = {errno.ENOTSUP, errno.EOPNOTSUPP}

# openx_np(2)'s properties of a new file for its mode and its ACL (sys/fcntl.h),
# the ACL flag that keeps a new file from taking its folder's entries
# (sys/acl.h), and pathconf(2)'s tests of whether a volume keeps ACLs and
# whether its server, rather than the Mac, applies them (sys/unistd.h).
_FILESEC_MODE = 4
_FILESEC_ACL = 5
_ACL_FLAG_NO_INHERIT = 1 << 17
_PC_EXTENDED_SECURITY_NP = 13
_PC_AUTH_OPAQUE_NP = 14

# How much of the staged contents is written over the file at a time.
_CHUNK = 1024 * 1024

# getattrlist(2) and setattrlist(2): the creation date and Finder's Date Added,
# and which of the asked-for attributes the volume returned (sys/attr.h). Each
# date is a struct timespec, seconds and nanoseconds of 64 bits each.
_ATTR_BIT_MAP_COUNT = 5
_ATTR_CMN_CRTIME = 0x00000200
_ATTR_CMN_ADDEDTIME = 0x10000000
_ATTR_CMN_RETURNED_ATTRS = 0x80000000
_TIMESPEC_SIZE = 16

# The name of a staged file (see _stage()), which carries its account's tag (see
# _account()) and then six random hex digits. The folders where one was left, as
# a folder that lets the user add files but not remove them leaves one (on
# Windows that can't be told before trying), where saves stage in the temp
# folder and write over the file, so no more are left there; and the folders
# already looked through for one an earlier session left (see
# _can_stage_beside()).
_STAGED_NAME = re.compile(r"\.dynaface-([0-9a-f]{6})[0-9a-f]{6}\.tmp\Z")
_STUCK_FOLDERS: Set[str] = set()
_CHECKED_FOLDERS: Set[str] = set()


class _AttrList(ctypes.Structure):
    """struct attrlist: the attributes getattrlist(2) reads or setattrlist(2)
    sets."""

    _fields_ = [
        ("bitmapcount", ctypes.c_ushort),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


class _Timespec(ctypes.Structure):
    """struct timespec: a time in seconds and nanoseconds."""

    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class IncompleteSaveError(OSError):
    """A save that failed while writing over the file itself, which may now be
    incomplete. Only a save that couldn't replace the file writes over it."""


class MissingFileError(IncompleteSaveError):
    """A save that failed once a replace that failed may have changed the file
    itself (see save_file()): with the file gone, or no longer as it was, the
    saved contents were kept in kept, the staged file; with the staged file gone
    from its name, as a replace reported failing that was done leaves it, kept
    is None, and the file may hold the saved contents."""

    kept: Optional[str] = None


class _KeepInPlace(OSError):
    """A new file can't take the saved file's place with all that writing over
    the file keeps."""


class _NewFile:
    """
    A file a save creates, the staged copy or a new file, worked on through the
    open file rather than by name. Someone else who can change its folder can
    rename it and put a link to another file in its place, and nothing done
    through the open file reaches that other file. Windows lets no one else
    rename or remove a file while it is open, and renames only a file no one
    holds open, so there the staged copy is closed to be renamed (release()),
    and opened again by name only while the name still names it. A WebDAV
    volume (macOS) renames and removes only a file no one holds open too,
    reporting one held open busy, so there a file is closed once that fails (see
    _rename() and _remove()), and opened again for reading alone (see
    rewind()). Contents staged in memory have no path.
    """

    def __init__(self, path: Optional[str], file: BinaryIO) -> None:
        self.path = path
        self.file = file
        # Its status, noted when Windows closes it to rename it.
        self._released: Optional[os.stat_result] = None
        # Set once a rename of it into the saved file's place is refused busy,
        # which a volume can do once it removed that file, and once one fails
        # otherwise, which may have removed it, or been done (see _rename() and
        # save_file()).
        self.rename_refused = False
        self.rename_unsure = False

    @property
    def folder(self) -> Optional[str]:
        return None if self.path is None else os.path.dirname(self.path)

    def finish(self) -> None:
        """Write out what was written to the file, to the disk."""
        self.file.flush()
        if self.path is not None:
            os.fsync(self.file.fileno())

    def release(self) -> None:
        """Close the file so that it can be renamed or removed where only a file
        no one holds open can be (see above), noting which file it is, to know
        it again by name."""
        self._released = os.fstat(self.file.fileno())
        self.close()

    def is_at(self, path: str, alone: bool = False) -> bool:
        """Whether path names this file (never one in memory), and when alone,
        whether no other name does. Never where the volume doesn't number its
        files (see _identity()): a replace that failed must not pass for one
        done. False too where path can't be looked at (see found_at())."""
        try:
            return self.found_at(path, alone)
        except OSError:
            return False

    def found_at(self, path: str, alone: bool = False) -> bool:
        """is_at(), raising OSError where path can't be looked at."""
        if self.path is None:
            return False
        if self.file.closed:
            # Closed to be renamed or removed (see release()).
            named = _status_through_file(path)
            held = None if self._released is None else _identity(self._released)
        else:
            named = _named_status(path)
            held = _identity(os.fstat(self.file.fileno()))
        return (
            held is not None
            and stat.S_ISREG(named.st_mode)
            and _identity(named) == held
            and not (alone and named.st_nlink != 1)
        )

    def rewind(self) -> BinaryIO:
        """
        The file, to read from the start. One closed to be renamed (see
        release()) is opened again by name, never through a link put in its
        place, nor waiting on a FIFO put there, and used only if that is still
        the same file, of the same size: nothing is read from, or later emptied
        of, another put in its place. Where the volume doesn't number its files,
        the same file is told only by its size, and a replace reported failing
        that was done shows as a save that failed. On Windows it is opened for
        writing too, so that it can be emptied should it be left (see
        _discard()), unless a scanner lets others only read it. Elsewhere it is
        opened for reading alone: a WebDAV volume (macOS) locks a file opened for
        writing, and locking one that a rename reported failing took off the
        server can make the server create it anew, empty, or leave the volume
        trying to read it until no file on it can be opened.
        """
        if self.file.closed and self.path is not None:
            file: BinaryIO
            if not _WINDOWS:
                file = open(self.path, "rb", opener=_reopen)
            else:
                try:
                    file = open(self.path, "r+b", opener=_reopen)
                except PermissionError:
                    file = open(self.path, "rb", opener=_reopen)
            try:
                same = self._was(os.fstat(file.fileno()))
            except OSError:
                same = False
            if not same:
                file.close()
                raise OSError(
                    errno.ESTALE,
                    "another file was put in the staged file's place",
                    self.path,
                )
            self.file = file
        self.file.seek(0)
        return self.file

    def _was(self, status: os.stat_result) -> bool:
        """Whether status, of a file opened by name, is this one's as it was
        closed (see release()): a file of the same size, and the same file where
        the volume numbers its files. Nothing changes the staged file's size
        once it is closed."""
        released = self._released
        if released is None:
            return False
        known = _identity(released)
        return (
            stat.S_ISREG(status.st_mode)
            and status.st_size == released.st_size
            and (known is None or _identity(status) == known)
        )

    def empty(self, by_name: bool) -> bool:
        """Empty the file and say whether that was done: through it, open, or
        where it is open for reading alone or closed (see rewind() and
        release()) and by_name, through it opened again by name for writing,
        while that is still the same file, which needs a volume that numbers its
        files (see _discard())."""
        if not self.file.closed:
            with contextlib.suppress(OSError):
                os.ftruncate(self.file.fileno(), 0)
                return True
        released = self._released
        if not by_name or self.path is None or released is None:
            return False
        if _identity(released) is None:
            return False
        try:
            with open(self.path, "r+b", opener=_reopen) as file:
                if not self._was(os.fstat(file.fileno())):
                    return False
                os.ftruncate(file.fileno(), 0)
                return True
        except OSError:
            return False

    def close(self) -> None:
        """Close the file. Whatever its buffer still holds is dropped: that
        happens only when the save failed, and the file is discarded then. A
        failure to close it is logged, not raised: what the file holds was
        written out before (see finish()), or else the save failed, and that
        error is the one to tell."""
        raw = getattr(self.file, "raw", None)
        try:
            if raw is not None:
                raw.close()
            self.file.close()
        except OSError:
            logger.warning(f"Unable to close {self.path}", exc_info=True)


def _identity(status: os.stat_result) -> Optional[Tuple[int, int]]:
    """Which file a status is of: its volume and its number there. None where
    the volume doesn't number its files: some Windows volumes give every file
    the number 0 (WebDAV, Google Drive), which tells no file from another."""
    if not status.st_ino:
        return None
    return status.st_dev, status.st_ino


def _named_status(path: str) -> os.stat_result:
    """The status of what path names, without following a link. Where an ACL
    denies reading it by name, but lets the file be opened, as macOS's can, it
    is read through the file."""
    try:
        return os.lstat(path)
    except PermissionError:
        return _status_through_file(path)


def _reopen(path: str, flags: int) -> int:
    """Open path, a file the save made and closed (see _NewFile.release()), by
    name again: never through a link put in its place, nor waiting on a FIFO
    put there."""
    fd = os.open(path, flags | _NOFOLLOW | _NONBLOCK)
    try:
        if _NONBLOCK:
            # Only for opening it: reading and writing it wait as usual.
            os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _status_through_file(path: str) -> os.stat_result:
    """The status of the file path names, read through it opened for reading,
    without following a link where the system can tell it not to (Windows
    can't). macOS tells an open file's attributes only when it was opened for
    reading."""
    fd = os.open(path, os.O_RDONLY | _BINARY | _NOFOLLOW | _NONBLOCK)
    try:
        return os.fstat(fd)
    finally:
        os.close(fd)


def save_file(filename: str, write: Callable[[BinaryIO], None]) -> None:
    """
    Save filename, calling write with the file to write its contents to.

    The contents go to a new file beside the old one, which then takes the old
    one's place in a single step, so a save that fails part way (a full disk, an
    error in write) leaves the old file as it was. The new file gets what writing
    the old one in place would have kept: its group and permissions, and on macOS
    its ACL, exactly, its extended attributes (Finder tags among them), its hidden
    and nodump flags, and its creation date and Date Added where the volume lets
    them be set. A file reached through a link is saved where the link points. On
    Windows the new file has its folder's permissions, as any new file there
    does, so a document given an ACL or an owner of its own loses them. On a
    WebDAV volume the server drops what it kept of the old file's properties, as
    Windows keeps a file's attributes and creation date there. A file that
    doesn't exist yet is written where it goes, as there is nothing to keep, and
    removed again if the save fails.

    A network volume can remove the old file and then fail to put the new one in
    its place: smbfs (macOS) replaces a file in those two steps, and so can a
    WebDAV server, even where the failure is a busy file (smbfs, when someone
    else holds the new one, and SabreDAV). A replace that fails is tried once
    more (see _rename()). A save that still fails, once a replace failed, raises
    MissingFileError where the old file is gone, no longer to be found, no
    longer a file or no longer of its size, keeping the new contents in the
    staged file. Where replaces were only refused busy, as a WebDAV lock refuses
    them, a new size counts only in another file, as the lock may be another
    computer's, which saved its own version (see _lost()). Once a replace failed
    otherwise, where the staged file is gone from its name, as a replace
    reported failing that was done leaves it, which can't always be told
    otherwise (a volume that numbers no files, and WebDAV's cache on macOS,
    which shows the old file for a while), it raises MissingFileError with
    nothing kept: the file may hold the new contents then. A save that never
    tried a replace says nothing of the kind, whatever the volume shows after.

    Where a new file can't take the old one's place with all of that, the
    finished contents are written over the old file instead, so an error in write
    still leaves it as it was, but a failure while writing over it leaves it
    incomplete, raising IncompleteSaveError. That happens for a file with other
    hard links, which would keep the old contents; one with an owner or group the
    new file can't have; one whose attributes, extended attributes or ACL can't
    be read or copied; on macOS, one the user can't remove (which replacing it
    does), as an ACL can deny and a read-only folder does; on Windows, a hidden or
    system file, or a compressed or encrypted one in a folder that doesn't
    compress or encrypt new files; and one in a folder that refuses the new file
    or the replacement. The contents are then staged beside the file where a
    staged file can be removed again, so they stay on its volume and under its
    folder's encryption, and in the temp folder otherwise. An encrypted Windows
    document is staged in memory where the staged file wouldn't be encrypted.

    The staged file is used through the open file, not by name (see _NewFile),
    and the old file only while it is still the file the save began with, so
    someone else who can change a shared folder can't make the save read, change
    or write into another file by putting a link to it in either one's place:
    the file is written over then, or the save refused. A file nothing can be
    known of to tell it by is opened when the save begins, as writing it in
    place did, and written over through that. They can still do what they could
    anyway, put another file in the document's place. Nor, on macOS, can an ACL
    entry the folder hands new files let them open the staged file (see
    _stage()), save for a moment on a network volume whose server applies the
    entry.

    A file that can't be written is refused with PermissionError, as writing it
    in place was.
    """
    target = os.path.realpath(filename)
    exists, old = _stat(target)
    if not exists:
        _create(target, write)
        return
    _check_writable(target, filename)
    # A file whose attributes can't be read is written over (nothing is known
    # of it to keep), through the file opened now.
    held = _open_to_write_over(target) if old is None else None
    try:
        folder = os.path.dirname(target)
        staged = _stage(folder if _can_stage_beside(target, folder) else None, old)
        placed = kept = False
        try:
            write(staged.file)
            staged.finish()
            if staged.folder == folder:
                placed = _replace(staged, target, old)
            if not placed:
                try:
                    _write_over(staged, target, old, held)
                except IncompleteSaveError:
                    raise
                except Exception as e:
                    missing = _missing(e, staged, target, old)
                    if missing is None:
                        raise
                    kept = missing.kept is not None
                    raise missing from e
        finally:
            try:
                if not (placed or kept):
                    _discard(staged, folder)
            finally:
                staged.close()
    except BaseException:
        if held is not None:
            _close_quietly(held)
        raise
    if held is not None:
        # It was written over.
        _close_written(held, target)


def _stat(target: str) -> Tuple[bool, Optional[os.stat_result]]:
    """Whether target exists, and its status. An ACL can deny reading a file's
    attributes by name while letting it be opened, and the open file tells them
    then (macOS). The status is None for a file whose attributes can't be read
    at all, which an ACL can deny while letting the file be written: nothing is
    known about such a file to keep, so it is written over (see save_file())."""
    try:
        return True, os.stat(target)
    except FileNotFoundError:
        return False, None
    except PermissionError:
        pass
    try:
        return True, _status_through_file(target)
    except FileNotFoundError:
        return False, None
    except OSError:
        return True, None


def _check_writable(target: str, filename: str) -> None:
    """Raise PermissionError for a file that can't be written."""
    if _WINDOWS:
        # Windows' os.access() reads only the read-only attribute; an ACL that
        # denies writing shows only when the file is opened for writing.
        os.close(os.open(target, os.O_WRONLY))
    elif not os.access(target, os.W_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), filename)


def _create(target: str, write: Callable[[BinaryIO], None]) -> None:
    """Write a new file at target, which is emptied and removed again when the
    save fails. It takes its permissions from the umask and its folder, as any
    new file does; it is opened for reading too, as macOS tells an open file's
    attributes only then, which an entry from its folder can deny reading by
    name. Raises FileExistsError when a file, or a link, has appeared at target
    since the save began."""
    created = _NewFile(target, open(target, "x+b"))
    try:
        try:
            write(created.file)
            created.finish()
        except BaseException:
            # Nothing was there before, so leave nothing.
            _discard(created, None)
            raise
    finally:
        created.close()


def _can_stage_beside(target: str, folder: str) -> bool:
    """
    Whether to stage the contents beside target. Not on macOS beside a file the
    user can't remove: replacing a file removes it, which an ACL on the file or
    its folder can deny. Nor in a folder where a staged file was left (see
    _discard()), which the file left records from one session to the next: the
    first save in a folder each session removes the empty staged files left
    there, and stages in the temp folder if one can't be removed.
    """
    if sys.platform == "darwin" and not os.access(target, _DELETE_OK):
        return False
    if folder not in _CHECKED_FOLDERS:
        _CHECKED_FOLDERS.add(folder)
        if not _remove_left_files(folder):
            _STUCK_FOLDERS.add(folder)
    return folder not in _STUCK_FOLDERS


def _remove_left_files(folder: str) -> bool:
    """
    Remove the empty staged files that earlier saves of the user's left in
    folder: those whose names carry the user's account's tag (see _account()),
    and where files have owners, that are the user's. False when one can't be
    removed. Another account's are left alone even where every file seems the
    user's (on Windows, and on a network volume mounted as the user): in a
    folder where each account may remove only its own files, failing to remove
    one tells nothing of the user's own.
    """
    try:
        with os.scandir(folder) as entries:
            left = [entry.path for entry in entries if _is_own_staged(entry.name)]
    except OSError:
        return True
    removed = True
    for path in left:
        try:
            status = os.lstat(path)
            if not stat.S_ISREG(status.st_mode) or status.st_size:
                continue
            if hasattr(os, "geteuid") and status.st_uid != os.geteuid():
                continue
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            removed = False
    return removed


def _stage(folder: Optional[str], old: Optional[os.stat_result]) -> _NewFile:
    """
    Create the file a save writes first: in folder, so it can take the saved
    file's place, or in the temp folder when there is no folder or it refuses the
    file. Its name never grows with the saved file's. It starts private, since it
    holds all of the file before it takes the file's permissions: readable by its
    owner alone and, on macOS, without the ACL entries its folder hands new files,
    which could let others read it or deny changing or removing it. Where the Mac
    applies those, it is created without them (see _private()); where a network
    volume's server does, they are taken off at once (see _make_private()), and
    where they can't be, it is created in the temp folder instead. A Windows file
    is encrypted only when it is created in a folder that encrypts new files, so
    an encrypted document is staged in memory where the staged file isn't
    encrypted, and no copy of it is written in the clear.
    """
    staged = None
    if folder is not None:
        path = os.path.join(folder, _staged_name())
        with contextlib.suppress(OSError):
            staged = _NewFile(path, open(path, "x+b", opener=_private))
        if staged is not None and not _make_private(staged):
            _discard(staged, folder)
            staged.close()
            staged = None
    if staged is None:
        # The user's own temp folder hands new files no ACL entries others could
        # use, as a rule, so one there is used even where they stay.
        path = os.path.join(tempfile.gettempdir(), _staged_name())
        staged = _NewFile(path, open(path, "x+b", opener=_private))
        _make_private(staged)
    if _loses_encryption(staged, old):
        _discard(staged, folder)
        staged.close()
        return _NewFile(None, io.BytesIO())
    return staged


def _staged_name() -> str:
    """A new staged file's name (see _STAGED_NAME), of 26 characters: kept that
    short because Windows, unless long paths are turned on, refuses a path of
    260 characters or more, and a staged file refused beside a document in a
    deep folder is written over it, not replaced."""
    return f".dynaface-{_account()}{secrets.token_hex(3)}.tmp"


def _is_own_staged(name: str) -> bool:
    """Whether name is that of a staged file of the user's account."""
    match = _STAGED_NAME.match(name)
    return match is not None and match.group(1) == _account()


@functools.lru_cache(maxsize=None)
def _account() -> str:
    """The tag of the user's account that their staged files' names carry, so a
    save tells its account's from others' (see _remove_left_files()): six hex
    digits of a hash of what the computer is known by (see _computer()) and the
    user name, which the names don't show. Accounts of one name on two
    computers, saving to a shared folder, so get different tags; what the user
    left there from another computer is left for that one to remove, as it may
    be writing it still, though the folder shows it empty (a network volume
    shows what another computer writes only once that one sends it)."""
    try:
        name = getpass.getuser()
    except Exception:
        # No user name to be found (getpass raises KeyError, OSError or
        # ImportError, by Python version and system).
        name = ""
    account = f"{_computer()}\0{name}"
    return hashlib.sha256(account.encode("utf-8", "replace")).hexdigest()[:6]


def _computer() -> str:
    """What the computer is known by in the tag (see _account()), which must
    stay the same from one session to the next: on macOS its hardware UUID, as
    a Mac's host name, unless the user set one, is whatever the network it is
    on calls its address; elsewhere, and where the Mac tells none or one of
    zeros, which tells no Mac from another, the host name, the computer's own
    name."""
    if sys.platform == "darwin":
        uuid = ctypes.create_string_buffer(16)
        with contextlib.suppress(OSError):
            # Waiting at most a second for it, which is known once the Mac has
            # started.
            told = not _libc().gethostuuid(uuid, ctypes.byref(_Timespec(1, 0)))
            if told and any(uuid.raw):
                return uuid.raw.hex()
    try:
        return socket.gethostname()
    except OSError:
        return ""


def _private(path: str, flags: int) -> int:
    """Open a new file readable and writable by its owner alone. On macOS,
    where the Mac hands new files the ACL entries their folder hands down, it is
    created without them, in one step: no one they would let open it can, as
    taking them off after would leave a moment to, and a file opened then stays
    open to them after."""
    if _hands_down_entries(os.path.dirname(path)):
        return _open_without_entries(path, flags)
    return os.open(path, flags, 0o600)


def _hands_down_entries(folder: str) -> bool:
    """Whether the Mac gives a new file in folder the ACL entries the folder
    hands down: not where the volume keeps no ACLs (FAT, exFAT), nor where its
    server applies them (a network volume)."""
    if sys.platform != "darwin":
        return False
    try:
        return bool(os.pathconf(folder, _PC_EXTENDED_SECURITY_NP)) and not (
            os.pathconf(folder, _PC_AUTH_OPAQUE_NP)
        )
    except (OSError, ValueError):
        return False


def _open_without_entries(path: str, flags: int) -> int:
    """Create path with openx_np(2), readable and writable by its owner alone
    and with an empty ACL flagged to take no entries from its folder, which
    leaves it with no ACL."""
    libc = _libc()
    security = libc.filesec_init()
    if not security:
        raise _libc_error(path)
    try:
        acl = libc.acl_init(0)
        if not acl:
            raise _libc_error(path)
        try:
            flagset = ctypes.c_void_p()
            mode = ctypes.c_uint16(0o600)
            if (
                libc.acl_get_flagset_np(acl, ctypes.byref(flagset))
                or libc.acl_add_flag_np(flagset, _ACL_FLAG_NO_INHERIT)
                or libc.acl_set_flagset_np(acl, flagset)
                or libc.filesec_set_property(
                    security, _FILESEC_MODE, ctypes.byref(mode)
                )
                # The filesec keeps a copy of the ACL.
                or libc.filesec_set_property(
                    security, _FILESEC_ACL, ctypes.byref(ctypes.c_void_p(acl))
                )
            ):
                raise _libc_error(path)
        finally:
            libc.acl_free(acl)
        fd = libc.openx_np(os.fsencode(path), flags, security)
        if fd < 0:
            raise _libc_error(path)
        return fd
    finally:
        libc.filesec_free(security)


def _make_private(staged: _NewFile) -> bool:
    """Take off the ACL entries a new staged file took from its folder (macOS),
    as a network volume's server gives them. False when it may still have
    some."""
    if sys.platform != "darwin" or staged.path is None:
        return True
    return _clear_acl(staged.file.fileno(), staged.path)


def _clear_acl(fd: int, path: str) -> bool:
    """
    Take off the ACL entries of an open file the user owns. False when it may
    still have some. The empty ACL is set without reading the file's first, as
    entries a folder hands out can deny its owner reading them, though never
    changing them. Where a volume refuses even that, as a network volume might,
    the file may have no entries to take off.
    """
    try:
        libc = _libc()
        empty = libc.acl_init(0)
        if not empty:
            raise _libc_error(path)
        try:
            if libc.acl_set_fd_np(fd, empty, _ACL_TYPE_EXTENDED) == 0:
                return True
            failure = _libc_error(path)
        finally:
            libc.acl_free(empty)
        if failure.errno in _NO_ACLS_KEPT:
            return True
        acl = _acl_or_none(libc.acl_get_fd_np(fd, _ACL_TYPE_EXTENDED), path)
        if acl is not None:
            libc.acl_free(acl)
            raise failure
        return True
    except OSError:
        logger.info(f"Unable to clear the ACL of {path}", exc_info=True)
        return False


def _loses_encryption(staged: _NewFile, old: Optional[os.stat_result]) -> bool:
    """Whether the staged file would hold an encrypted file in the clear
    (Windows)."""
    encrypted = stat.FILE_ATTRIBUTE_ENCRYPTED
    return bool(
        old is not None
        and getattr(old, "st_file_attributes", 0) & encrypted
        and not _file_attributes(staged) & encrypted
    )


def _file_attributes(staged: _NewFile) -> int:
    """The staged file's Windows file attributes (none elsewhere)."""
    return getattr(os.fstat(staged.file.fileno()), "st_file_attributes", 0)


def _replace(staged: _NewFile, target: str, old: Optional[os.stat_result]) -> bool:
    """
    Put the staged file in target's place, with target's metadata. False when
    that can't be done, with target as it was, unless a rename that failed
    removed it, or was done where that can't be told (see _rename() and
    save_file()).
    """
    if staged.path is None:
        return False
    try:
        _take_metadata(staged, target, old)
        if not (_WINDOWS or staged.is_at(staged.path, alone=True)):
            raise _KeepInPlace("another file was put in the staged file's place")
        if _WINDOWS:
            staged.release()
        _rename(staged, staged.path, target)
    except _KeepInPlace as e:
        logger.info(f"Writing over {target}: {e}")
        return False
    except OSError:
        if not staged.is_at(target):
            logger.info(f"Unable to replace {target}; writing over it", exc_info=True)
            return False
        # A network volume can report a rename failing that it did.
        logger.info(f"Replaced {target}, though that was reported failing")
    _restore_flags(staged, target, old)
    return True


def _rename(staged: _NewFile, path: str, target: str) -> None:
    """
    Rename the staged file, at path, to target. One that fails is tried once
    more while the staged file's name still names it alone, as the save checked
    before: a network volume can remove target and then fail to rename the
    staged file (smbfs removes it first, and so can a WebDAV server answering
    500), and target, once removed, needs no removing. Where the volume renames
    no file held open, reporting it busy, as a WebDAV volume does (macOS), the
    staged file is closed (see _NewFile.release()) and renamed by name, which
    isn't counted as trying again. A rename reported failing that was done,
    which leaves nothing at path, is left for _replace() to tell, and where it
    can't, for save_file(). Each failure is noted (see _NewFile): smbfs removes
    target before it reports busy a staged file someone else holds, as SabreDAV
    does before it reports a file it holds locked, and a failure of another
    kind may have been done.
    """
    rename = functools.partial(_retry, os.replace, path, target)
    again = True
    while True:
        try:
            rename()
            return
        except OSError as e:
            failure = e
        busy = failure.errno == errno.EBUSY
        if busy:
            staged.rename_refused = True
        else:
            staged.rename_unsure = True
        if busy and not staged.file.closed:
            staged.release()
        elif again:
            again = False
        else:
            raise failure
        if not staged.is_at(path, alone=True):
            raise failure
        rename = functools.partial(os.replace, path, target)


def _take_metadata(
    staged: _NewFile, target: str, old: Optional[os.stat_result]
) -> None:
    """
    Give the staged file what writing over target would keep: its group, then on
    macOS its flags, dates, extended attributes and ACL, then its permissions,
    last, so that the staged file stays private until it has the ACL as well.
    Raises _KeepInPlace when the staged file can't have all of it, and another
    OSError when a step fails.
    """
    if old is None:
        raise _KeepInPlace("its attributes can't be read")
    if old.st_nlink > 1:
        raise _KeepInPlace("its other hard links would keep the old contents")
    if hasattr(os, "geteuid") and old.st_uid != os.geteuid():
        # Only an administrator can give a file to another account.
        raise _KeepInPlace("it belongs to another account")
    attributes = getattr(old, "st_file_attributes", 0)
    if attributes & _NEVER_NEW:
        raise _KeepInPlace("it is hidden or a system file")
    if (attributes ^ _file_attributes(staged)) & _FROM_FOLDER:
        raise _KeepInPlace("it is compressed or encrypted unlike a new file there")
    fd = staged.file.fileno()
    if hasattr(os, "geteuid") and old.st_gid != os.fstat(fd).st_gid:
        # Fails for a group the user isn't in.
        os.chown(fd, -1, old.st_gid)
    if sys.platform == "darwin":
        _take_mac_metadata(fd, target, old)
    mode = stat.S_IMODE(old.st_mode)
    if os.chmod in os.supports_fd:
        os.chmod(fd, mode)
    elif staged.path is not None:
        # Windows, where no one else can rename the file while it is open.
        os.chmod(staged.path, mode)


def _take_mac_metadata(fd: int, target: str, old: os.stat_result) -> None:
    """
    Give the staged file (fd) target's hidden and nodump flags, dates, extended
    attributes and ACL, read through target opened without following a link,
    and only while it is the file the save began with. The ACL comes last, as it
    can deny changing the staged file. Raises OSError when a step other than the
    dates fails.
    """
    libc = _libc()
    source = os.open(target, os.O_RDONLY | _NOFOLLOW | _NONBLOCK)
    try:
        status = os.fstat(source)
        if (status.st_dev, status.st_ino) != (old.st_dev, old.st_ino):
            raise _KeepInPlace("another file took its place during the save")
        flags = status.st_flags & (stat.UF_HIDDEN | stat.UF_NODUMP)
        if flags and libc.fchflags(fd, os.fstat(fd).st_flags | flags):
            raise _libc_error(target)
        _copy_dates(source, fd, target)
        if libc.fcopyfile(source, fd, None, _COPYFILE_XATTR):
            raise _libc_error(target)
        _copy_acl(source, fd, target)
    finally:
        os.close(source)


def _copy_dates(source: int, fd: int, target: str) -> None:
    """Give the staged file (fd) target's (source's) creation date and Date
    Added, as far as its volume keeps them and lets them be set. A new file has
    the time of the save in both; one that can't be copied stays so."""
    try:
        libc = _libc()
        asked = _AttrList(
            _ATTR_BIT_MAP_COUNT,
            0,
            _ATTR_CMN_RETURNED_ATTRS | _ATTR_CMN_CRTIME | _ATTR_CMN_ADDEDTIME,
            0,
            0,
            0,
            0,
        )
        buffer = ctypes.create_string_buffer(64)
        if libc.fgetattrlist(source, ctypes.byref(asked), buffer, len(buffer), 0):
            raise _libc_error(target)
        # The length of what was returned, then which attributes were (five
        # groups of flags, the common ones first), then each date returned.
        length, returned = struct.unpack_from("=II", buffer.raw)
        offset = 4 + 4 * _ATTR_BIT_MAP_COUNT
        for attribute in (_ATTR_CMN_CRTIME, _ATTR_CMN_ADDEDTIME):
            if not returned & attribute or offset + _TIMESPEC_SIZE > length:
                continue
            value = buffer.raw[offset : offset + _TIMESPEC_SIZE]
            offset += _TIMESPEC_SIZE
            setting = _AttrList(_ATTR_BIT_MAP_COUNT, 0, attribute, 0, 0, 0, 0)
            data = ctypes.create_string_buffer(value, len(value))
            if libc.fsetattrlist(fd, ctypes.byref(setting), data, len(value), 0):
                raise _libc_error(target)
    except OSError:
        logger.info(f"Unable to copy the dates of {target}", exc_info=True)


def _copy_acl(source: int, fd: int, target: str) -> None:
    """Give the staged file (fd) exactly target's (source's) ACL. (copyfile(3)
    would keep the entries the staged file took from its folder and drop the
    ones target took from its own.) The staged file was made without any, so
    nothing is set when target has none, as on a volume that keeps no ACLs."""
    libc = _libc()
    acl = _acl_or_none(libc.acl_get_fd_np(source, _ACL_TYPE_EXTENDED), target)
    if acl is None:
        return
    try:
        if libc.acl_set_fd_np(fd, acl, _ACL_TYPE_EXTENDED):
            raise _libc_error(target)
    finally:
        libc.acl_free(acl)


def _acl_or_none(acl: Any, filename: str) -> Any:
    """What acl_get_fd_np() returned for filename: its ACL, or None when it has
    none."""
    if acl:
        return acl
    code = ctypes.get_errno()
    if code in _NO_ACL:
        return None
    raise OSError(code, os.strerror(code), filename)


def _restore_flags(
    staged: _NewFile, target: str, old: Optional[os.stat_result]
) -> None:
    """Give the replacement the hidden and nodump flags the old file had, where
    the replace lost them: FAT and exFAT volumes take a file's hidden flag from
    its name."""
    wanted = getattr(old, "st_flags", 0) & (stat.UF_HIDDEN | stat.UF_NODUMP)
    if not wanted or staged.file.closed:
        return
    try:
        fd = staged.file.fileno()
        flags = os.fstat(fd).st_flags
        if flags & wanted != wanted and _libc().fchflags(fd, flags | wanted):
            raise _libc_error(target)
    except OSError:
        logger.info(f"Unable to restore the flags of {target}", exc_info=True)


@functools.lru_cache(maxsize=None)
def _libc() -> Any:
    """The C library's calls on open files for flags, attributes, extended
    attributes and ACLs, for creating a file without its folder's ACL entries,
    and for the Mac's hardware UUID (macOS). Raises OSError where it lacks
    one."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.fchflags.argtypes = [ctypes.c_int, ctypes.c_uint32]
        libc.fchflags.restype = ctypes.c_int
        libc.fcopyfile.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        libc.fcopyfile.restype = ctypes.c_int
        for function in (libc.fgetattrlist, libc.fsetattrlist):
            function.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(_AttrList),
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
        libc.acl_init.argtypes = [ctypes.c_int]
        libc.acl_init.restype = ctypes.c_void_p
        libc.acl_free.argtypes = [ctypes.c_void_p]
        libc.acl_free.restype = ctypes.c_int
        libc.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
        libc.acl_get_fd_np.restype = ctypes.c_void_p
        libc.acl_set_fd_np.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        libc.acl_set_fd_np.restype = ctypes.c_int
        libc.acl_get_flagset_np.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        libc.acl_get_flagset_np.restype = ctypes.c_int
        libc.acl_add_flag_np.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libc.acl_add_flag_np.restype = ctypes.c_int
        libc.acl_set_flagset_np.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libc.acl_set_flagset_np.restype = ctypes.c_int
        libc.filesec_init.argtypes = []
        libc.filesec_init.restype = ctypes.c_void_p
        libc.filesec_free.argtypes = [ctypes.c_void_p]
        libc.filesec_free.restype = None
        libc.filesec_set_property.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        libc.filesec_set_property.restype = ctypes.c_int
        libc.openx_np.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
        libc.openx_np.restype = ctypes.c_int
        libc.gethostuuid.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Timespec)]
        libc.gethostuuid.restype = ctypes.c_int
    except AttributeError as e:
        raise OSError(errno.ENOSYS, str(e)) from e
    return libc


def _libc_error(filename: str) -> OSError:
    """The OSError for the C library call that just failed."""
    code = ctypes.get_errno()
    return OSError(code, os.strerror(code), filename)


def _write_over(
    staged: _NewFile,
    target: str,
    old: Optional[os.stat_result],
    held: Optional[int] = None,
) -> None:
    """
    Write the staged contents over target, keeping everything else about it:
    through held, target opened when the save began, where it was. Raises
    IncompleteSaveError when this fails after it began changing target.
    """
    source = staged.rewind()
    # Refuse now, rather than part way, contents that can't fit in the free
    # room and the room the old contents take, where that is known.
    size = source.seek(0, os.SEEK_END)
    free = _free_room(os.path.dirname(target))
    reused = 0 if old is None else old.st_size
    if free is not None and size > reused + free:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), target)
    if held is not None:
        # save_file() closes it.
        _write_into(source, size, held, target)
        return
    fd = _open_known(target, old)
    try:
        _write_into(source, size, fd, target)
    except BaseException:
        _close_quietly(fd)
        raise
    _close_written(fd, target)


def _free_room(folder: str) -> Optional[int]:
    """The room free on folder's volume, or None where that isn't known: where
    it can't be read, and where the volume tells no size, as a WebDAV volume
    whose server keeps no quota tells none (macOS), and shows no room free."""
    try:
        usage = shutil.disk_usage(folder)
    except OSError:
        return None
    return usage.free if usage.total else None


def _open_known(target: str, old: Optional[os.stat_result]) -> int:
    """Open target to write over it (see _open_to_write_over()), truncating it
    at the end, not now. A link put in its place since the save began is
    refused rather than written through, and so, where that can be told (not
    on Windows), is any other file."""
    fd = _open_to_write_over(target)
    try:
        if old is not None and not _WINDOWS:
            status = os.fstat(fd)
            if (status.st_dev, status.st_ino) != (old.st_dev, old.st_ino):
                raise OSError(
                    errno.ESTALE, "another file took its place during the save", target
                )
    except BaseException:
        _close_quietly(fd)
        raise
    return fd


def _write_into(source: BinaryIO, size: int, fd: int, target: str) -> None:
    """
    Write source's size bytes over target, open as fd. The part past the file's
    end goes first, as only it needs room the volume may not have, though it
    reports the room free (APFS keeps some back): should that fail, the part is
    given back, leaving the file as it was, and the error raised as it came.
    Raises IncompleteSaveError once the file may have been changed.
    """
    try:
        # Told even where the file's attributes can't be read (macOS).
        length: Optional[int] = os.lseek(fd, 0, os.SEEK_END)
    except OSError:
        length = None
    head = size
    if length is not None and size > length:
        try:
            _copy_range(source, fd, length, size)
            os.fsync(fd)
        except BaseException as e:
            try:
                os.ftruncate(fd, length)
            except OSError:
                raise _incomplete(e, target) from e
            raise
        head = length
    try:
        _copy_range(source, fd, 0, head)
        os.ftruncate(fd, size)
        os.fsync(fd)
    except Exception as e:
        raise _incomplete(e, target) from e


def _copy_range(source: BinaryIO, fd: int, start: int, end: int) -> None:
    """Copy source's bytes from start to end to the same place in the file open
    as fd."""
    source.seek(start)
    os.lseek(fd, start, os.SEEK_SET)
    left = end - start
    while left > 0:
        chunk = source.read(min(left, _CHUNK))
        if not chunk:
            raise OSError(errno.EIO, "the staged contents ended early")
        view = memoryview(chunk)
        while view:
            written = os.write(fd, view)
            if not written:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            view = view[written:]
        left -= len(chunk)


def _incomplete(error: BaseException, target: str) -> IncompleteSaveError:
    """The IncompleteSaveError for an error that came while writing over
    target."""
    return IncompleteSaveError(getattr(error, "errno", None), _reason(error), target)


def _missing(
    error: Exception, staged: _NewFile, target: str, old: Optional[os.stat_result]
) -> Optional[MissingFileError]:
    """
    The MissingFileError for error, which came while writing over target, or None
    for error to be raised as it came: only a rename that failed can have changed
    target (see _rename()). A staged file still at its name is kept where target
    may have lost the file the save began with (see _lost()). Once a rename
    failed otherwise than busy, one gone from its name may be in target's place,
    a rename reported failing that was done, and nothing is kept; a busy refusal
    is never one done, and a volume that stopped answering shows no file, so
    after busy refusals alone that tells nothing.
    """
    if staged.path is None or not (staged.rename_refused or staged.rename_unsure):
        return None
    kept: Optional[str] = None
    if os.path.lexists(staged.path):
        if old is None or not _lost(target, old, not staged.rename_unsure):
            return None
        kept = staged.path
    elif not staged.rename_unsure:
        return None
    missing = MissingFileError(getattr(error, "errno", None), _reason(error), target)
    missing.kept = kept
    return missing


def _reason(error: BaseException) -> str:
    """What went wrong, in an error's own words."""
    return getattr(error, "strerror", None) or str(error) or type(error).__name__


def _lost(target: str, old: os.stat_result, refused: bool) -> bool:
    """Whether target may have lost the file the save began with (old), which a
    write over it that failed left as it was: target is gone or can't be looked
    at, isn't a file, or isn't of the old file's size, as when a WebDAV server
    puts an empty file in the place of one a rename removed, once it is locked
    to be written over. Where renames were only refused busy (refused), another
    size counts only in another file: the lock refusing them may be another
    computer's, which saved its own version, and macOS's WebDAV volume shows
    that under the old file's number. Another number alone never counts: that
    volume gives an untouched file a new one when the server fails a request to
    lock or read it."""
    try:
        status = _named_status(target)
    except OSError:
        return True
    if not stat.S_ISREG(status.st_mode):
        return True
    if status.st_size == old.st_size:
        return False
    if not refused:
        return True
    known = _identity(old)
    return known is None or _identity(status) != known


def _close_quietly(fd: int) -> None:
    """Close the open file fd while an error is raised, which a failure to
    close it mustn't take the place of."""
    with contextlib.suppress(OSError):
        os.close(fd)


def _close_written(fd: int, target: str) -> None:
    """Close target, open as fd, once it was written over. A failure then, as a
    network volume can report for a write that failed, means it may be
    incomplete."""
    try:
        os.close(fd)
    except OSError as e:
        raise _incomplete(e, target) from e


def _open_to_write_over(target: str) -> int:
    """Open target to write over it, without following a link: for reading and
    writing where it can be, as macOS tells an open file's attributes only when
    it was opened for reading, and for writing alone otherwise, so that a file
    the user may write but not read is saved as well."""
    flags = _BINARY | _NOFOLLOW
    try:
        return os.open(target, os.O_RDWR | flags)
    except PermissionError:
        return os.open(target, os.O_WRONLY | flags)


def _discard(new: _NewFile, folder: Optional[str]) -> None:
    """
    Empty a file the save created and remove it, so that no copy of the document
    stays behind. Elsewhere than on Windows that is done only while the file's
    name still names it: one moved since, by someone else who can change the
    folder, or into the document's place by a replace a network volume reported
    failing, is left as it is, with no more than the document's permissions.
    (FAT and exFAT renumber a file they rename, so where it went can't be
    told.) It is emptied through the open file. One open for reading alone or
    closed (see _NewFile.rewind() and _NewFile.release()) is emptied through it
    opened again by name, while that is still the same file, and only once it
    is known to be there: when other links would keep its contents, or when its
    removal is refused, as opening a file for writing where a WebDAV volume no
    longer has it can make the server create it anew (see _NewFile.rewind()).
    An ACL entry the file took from its folder that denies removing it is taken
    off, as its owner always may, and where the volume removes no file held
    open (WebDAV, see _remove()), the file is closed to be removed. Windows
    removes only a file no one holds open, so there it is emptied through the
    open file, while no one else can rename it, then closed and removed by
    name: if it was closed already, to be renamed, whatever has its name now is
    only removed, never emptied. One that can't be removed (a folder that lets
    the user add files but not remove them, or a scanner holding it) is left
    rather than fail the save, empty unless it couldn't be emptied either (a
    scanner that lets others only read it); on macOS and Linux its name, which
    starts with a dot, hides it. When that happens in folder, saves there stage
    in the temp folder from then on (see _can_stage_beside()).
    """
    if new.path is None:
        return
    try:
        if _WINDOWS:
            if not new.file.closed:
                with contextlib.suppress(OSError):
                    os.ftruncate(new.file.fileno(), 0)
            new.close()
            _retry(os.remove, new.path)
        elif new.is_at(new.path):
            emptied = new.empty(by_name=_links(new, new.path) > 1)
            try:
                if new.file.closed:
                    os.remove(new.path)
                else:
                    _remove(new, new.path)
            except FileNotFoundError:
                raise
            except OSError:
                if not emptied:
                    new.empty(by_name=True)
                raise
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning(f"Unable to remove {new.path}", exc_info=True)
        if folder is not None and new.folder == folder:
            _STUCK_FOLDERS.add(folder)


def _links(new: _NewFile, path: str) -> int:
    """How many names new, at path, has: one where that can't be told."""
    try:
        if new.file.closed:
            return _named_status(path).st_nlink
        return os.fstat(new.file.fileno()).st_nlink
    except OSError:
        return 1


def _remove(new: _NewFile, path: str) -> None:
    """Remove path, which names new, open. On macOS an ACL entry the file took
    from its folder can deny that, and its owner can always take it off. Where
    the volume removes no file held open, reporting it busy, as a WebDAV volume
    does (macOS), it is closed first (see _NewFile.release()) and removed only
    while its name still names it; where that can't be told, the refusal is
    raised, so that it is emptied by name (see _discard())."""
    try:
        os.remove(path)
    except PermissionError:
        if sys.platform != "darwin" or not _clear_acl(new.file.fileno(), path):
            raise
        os.remove(path)
    except OSError as e:
        if e.errno != errno.EBUSY:
            raise
        new.release()
        try:
            there = new.found_at(path)
        except FileNotFoundError:
            return
        except OSError:
            raise e
        if there:
            os.remove(path)


def _retry(action: Callable[..., None], *args: str) -> None:
    """Run a file operation, trying again for about a second on Windows while it
    fails with PermissionError."""
    attempts = _ATTEMPTS if _WINDOWS else 1
    for attempt in range(attempts):
        try:
            action(*args)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(_RETRY_DELAY)
