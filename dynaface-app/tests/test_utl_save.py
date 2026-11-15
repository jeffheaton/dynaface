"""
Tests for utl_save.save_file(), which saves a file so that a save that fails part
way leaves the old file as it was.
"""

import builtins
import contextlib
import ctypes
import errno
import faulthandler
import io
import os
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import utl_save

ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
MACOS = sys.platform == "darwin"


def _writer(data):
    return lambda file: file.write(data)


def _fail(file):
    file.write(b"part of the document")
    raise ValueError("failed")


def _left_name(digits, account=None):
    """The name of a staged file an earlier save left (six hex digits), of the
    user's account unless another's is given."""
    return f".dynaface-{account or utl_save._account()}{digits}.tmp"


def _another_account():
    """The tag of an account other than the user's."""
    return "000000" if utl_save._account() != "000000" else "111111"


def _add_acl_entry(path, entry):
    subprocess.run(["/bin/chmod", "+a", entry, path], check=True)


def _acl(path):
    """path's ACL entries, as ls -le lists them."""
    listing = subprocess.run(
        ["/bin/ls", "-led", path], capture_output=True, text=True, check=True
    ).stdout
    return [line.strip() for line in listing.splitlines()[1:]]


class _Attributes(ctypes.Structure):
    _fields_ = [
        ("bitmapcount", ctypes.c_ushort),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


_ADDED_TIME = 0x10000000


def _date_added(path):
    """Finder's Date Added for path, in seconds and nanoseconds."""
    libc = ctypes.CDLL(None, use_errno=True)
    asked = _Attributes(5, 0, _ADDED_TIME, 0, 0, 0, 0)
    buffer = ctypes.create_string_buffer(64)
    if libc.getattrlist(os.fsencode(path), ctypes.byref(asked), buffer, 64, 0):
        raise OSError(ctypes.get_errno(), "getattrlist", path)
    return struct.unpack_from("=qq", buffer.raw, 4)


def _set_date_added(path, seconds):
    libc = ctypes.CDLL(None, use_errno=True)
    setting = _Attributes(5, 0, _ADDED_TIME, 0, 0, 0, 0)
    value = ctypes.create_string_buffer(struct.pack("=qq", seconds, 0), 16)
    if libc.setattrlist(os.fsencode(path), ctypes.byref(setting), value, 16, 0):
        raise OSError(ctypes.get_errno(), "setattrlist", path)


class _SaveTest(unittest.TestCase):
    """A document's folder, another folder of the user's, and the temp folder,
    with what the tests look at in them."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.folder = os.path.realpath(self._dir.name)
        self.path = os.path.join(self.folder, "doc.dyfc")
        # Another folder of the user's, for their files outside the document's.
        self._other = tempfile.TemporaryDirectory()
        self.other = os.path.realpath(self._other.name)
        # Where a staged file goes when the folder refuses it, so the tests see
        # that none is left there either.
        self._temp = tempfile.TemporaryDirectory()
        self.temp = os.path.realpath(self._temp.name)
        patcher = mock.patch.object(
            utl_save.tempfile, "gettempdir", return_value=self.temp
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        # The folders where a staged file was left, and those looked through for
        # one, which the module remembers for the session.
        for folders in (utl_save._STUCK_FOLDERS, utl_save._CHECKED_FOLDERS):
            folders.clear()
            self.addCleanup(folders.clear)
        # The account's tag, which a test that saves as on Windows would
        # otherwise leave for the tests after it.
        utl_save._account.cache_clear()
        self.addCleanup(utl_save._account.cache_clear)

    def tearDown(self):
        os.chmod(self.folder, 0o700)
        if MACOS:
            # The ACLs and flags tests give files keep them from being removed.
            for folder in (self.folder, self.other):
                subprocess.run(["/bin/chmod", "-R", "-N", folder])
                subprocess.run(["/usr/bin/chflags", "-R", "0", folder])
        self._dir.cleanup()
        self._temp.cleanup()
        self._other.cleanup()

    def save(self, data, path=None):
        utl_save.save_file(path or self.path, _writer(data))

    def read(self, path=None):
        with open(path or self.path, "rb") as f:
            return f.read()

    def inode(self):
        return os.stat(self.path).st_ino

    def assert_nothing_left(self, *others):
        """Asserts that the folder holds the file and others alone, and the temp
        folder nothing: no staged file was left."""
        self.assertEqual(
            sorted(os.listdir(self.folder)), sorted(("doc.dyfc",) + others)
        )
        self.assertEqual(os.listdir(self.temp), [])

    def private_file(self):
        """A file of the user's outside the document's folder, readable by them
        alone (where files have such permissions)."""
        path = os.path.join(self.other, "private")
        with open(path, "wb") as f:
            f.write(b"private")
        os.chmod(path, 0o600)
        status = os.stat(path)
        self._private_mode = status.st_mode
        self._private_group = status.st_gid
        return path

    def assert_untouched(self, private):
        """Asserts that private_file() is as it was made."""
        self.assertEqual(self.read(private), b"private")
        status = os.stat(private)
        self.assertEqual(status.st_mode, self._private_mode)
        self.assertEqual(status.st_gid, self._private_group)
        if MACOS:
            self.assertEqual(status.st_flags, 0)
            self.assertEqual(_acl(private), [])

    def give_another_group(self, path):
        """Give path a group of the user's other than its own. False where that
        can't be done: the user is in one group, or the volume ignores
        ownership."""
        if not hasattr(os, "getgroups"):
            return False
        group = os.stat(path).st_gid
        others = [g for g in os.getgroups() if g != group]
        if not others:
            return False
        try:
            os.chown(path, -1, others[0])
        except PermissionError:
            return False
        return os.stat(path).st_gid == others[0]

    @contextlib.contextmanager
    def no_waiting(self, seconds=5):
        """Fails what runs within, rather than wait forever, should it wait on a
        FIFO, however often it waits; where there is no SIGALRM (Windows), the
        test run is ended instead, with each thread's traceback, unless there is
        no stream to write them to (a runner that captures the output)."""
        if not hasattr(signal, "SIGALRM"):
            try:
                faulthandler.dump_traceback_later(
                    seconds, exit=True, file=sys.__stderr__
                )
                watched = True
            except (AttributeError, OSError, RuntimeError, ValueError):
                watched = False
            try:
                yield
            finally:
                if watched:
                    faulthandler.cancel_dump_traceback_later()
            return

        def waited(signum, frame):
            raise _Waited(f"waited more than {seconds} seconds")

        previous = signal.signal(signal.SIGALRM, waited)
        signal.setitimer(signal.ITIMER_REAL, seconds, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    @contextlib.contextmanager
    def renames_no_open_file(self, replace=os.replace, remove=os.remove):
        """Saves as on a WebDAV volume (macOS), which renames and removes no file
        held open, reporting it busy until the last file open on it is closed
        (POSIX, which lists the open files in /dev/fd). replace renames, and
        remove removes, a file no one holds open."""
        real_remove = remove

        def held_open(path):
            try:
                named = os.lstat(path)
            except OSError:
                return False
            for fd in os.listdir("/dev/fd"):
                with contextlib.suppress(OSError):
                    status = os.fstat(int(fd))
                    if (status.st_dev, status.st_ino) == (named.st_dev, named.st_ino):
                        return True
            return False

        def busy(action):
            def refused_while_open(path, *args):
                if held_open(path):
                    raise OSError(errno.EBUSY, "Resource busy", path)
                return action(path, *args)

            return refused_while_open

        with mock.patch.object(
            utl_save.os, "replace", busy(replace)
        ), mock.patch.object(utl_save.os, "remove", busy(real_remove)):
            yield


class _Waited(Exception):
    """What no_waiting() raises."""


class SaveFileTest(_SaveTest):
    def test_saves_a_new_file(self):
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left()

    def test_writes_a_new_file_where_it_goes(self):
        # A new file has nothing to keep, so no staged file is made for it.
        seen = []

        def write(file):
            seen.append((file.name, os.listdir(self.folder)))
            file.write(b"new")

        utl_save.save_file(self.path, write)
        self.assertEqual(seen, [(self.path, ["doc.dyfc"])])
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left()

    def test_a_failed_new_file_leaves_nothing(self):
        def fail(file):
            file.write(b"part")
            raise ValueError("failed")

        with self.assertRaises(ValueError):
            utl_save.save_file(self.path, fail)
        self.assertEqual(os.listdir(self.folder), [])
        self.assertEqual(os.listdir(self.temp), [])

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_a_failed_new_file_leaves_nothing_where_new_files_deny_changing_them(
        self,
    ):
        # The new file takes entries from the folder that deny emptying or
        # removing it by name: it is emptied through the open file, and its owner
        # takes the entries off to remove it.
        _add_acl_entry(
            self.folder, "everyone deny write,delete,file_inherit,only_inherit"
        )

        def fail(file):
            file.write(b"part")
            raise ValueError("failed")

        with self.assertRaises(ValueError):
            utl_save.save_file(self.path, fail)
        self.assertEqual(os.listdir(self.folder), [])
        self.assertEqual(os.listdir(self.temp), [])

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_a_failed_new_file_leaves_nothing_where_new_files_deny_reading_them(
        self,
    ):
        # The new file takes entries from the folder that deny reading its
        # attributes by name; it is known by them read through the open file.
        for denied in ("readattr", "readsecurity"):
            with self.subTest(denied):
                _add_acl_entry(
                    self.folder, f"everyone deny {denied},file_inherit,only_inherit"
                )
                try:
                    with self.assertRaises(ValueError):
                        utl_save.save_file(self.path, _fail)
                    self.assertEqual(os.listdir(self.folder), [])
                    self.assertEqual(os.listdir(self.temp), [])
                finally:
                    subprocess.run(["/bin/chmod", "-N", self.folder], check=True)

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_refuses_a_link_that_appears_at_a_new_file_s_name(self):
        # A link to a private file of the user's, put at the new document's name
        # after the save found nothing there, is refused, not written through.
        private = self.private_file()
        real_stat = utl_save._stat

        def link_appears(target):
            found = real_stat(target)
            os.symlink(private, target)
            return found

        with mock.patch.object(utl_save, "_stat", link_appears):
            with self.assertRaises(FileExistsError):
                self.save(b"new")
        self.assert_untouched(private)
        self.assertEqual(os.readlink(self.path), private)
        self.assertEqual(os.listdir(self.temp), [])

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_a_new_file_takes_its_permissions_from_the_umask(self):
        umask = os.umask(0)
        os.umask(umask)
        self.save(b"new")
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o666 & ~umask)

    def test_replaces_a_file_with_a_new_one(self):
        self.save(b"old")
        inode = self.inode()
        with self.assertNoLogs("utl_save", "WARNING"):
            self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertNotEqual(self.inode(), inode)
        self.assert_nothing_left()

    def test_a_failed_write_leaves_the_file_as_it_was(self):
        self.save(b"old")

        def fail(file):
            file.write(b"part")
            raise ValueError("failed")

        with self.assertRaises(ValueError):
            utl_save.save_file(self.path, fail)
        self.assertEqual(self.read(), b"old")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_a_staged_file_is_private_until_it_takes_the_permissions(self):
        # Whether it is made without the entries its folder hands down (macOS)
        # or as any other file is (elsewhere, and on a volume whose server
        # applies them, or that keeps no ACLs).
        for handed_down in (True, False) if MACOS else (False,):
            with self.subTest(handed_down=handed_down):
                self.save(b"old")
                os.chmod(self.path, 0o640)
                modes = []

                def write(file):
                    modes.append(os.fstat(file.fileno()).st_mode & 0o777)
                    file.write(b"new")

                with mock.patch.object(
                    utl_save, "_hands_down_entries", return_value=handed_down
                ):
                    utl_save.save_file(self.path, write)
                self.assertEqual(modes, [0o600])
                self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o640)

    @unittest.skipUnless(hasattr(os, "getgroups"), "POSIX groups")
    def test_keeps_the_group(self):
        self.save(b"old")
        if not self.give_another_group(self.path):
            self.skipTest("can't give the file another group here")
        group = os.stat(self.path).st_gid
        self.save(b"new")
        self.assertEqual(os.stat(self.path).st_gid, group)
        self.assertEqual(self.read(), b"new")

    def test_writes_over_a_file_with_other_hard_links(self):
        self.save(b"old")
        other = os.path.join(self.folder, "other.dyfc")
        os.link(self.path, other)
        inode = self.inode()
        staged_in = []

        def write(file):
            # The contents are staged beside the file, on its volume and under
            # its folder's encryption, not in the temp folder.
            staged_in.append(os.path.dirname(file.name))
            file.write(b"new")

        with self.assertNoLogs("utl_save", "WARNING"):
            utl_save.save_file(self.path, write)
        self.assertEqual(staged_in, [self.folder])
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.read(other), b"new")
        self.assertEqual(self.inode(), inode)
        self.assert_nothing_left("other.dyfc")

    @unittest.skipUnless(hasattr(os, "geteuid"), "POSIX owners")
    def test_writes_over_a_file_owned_by_another_account(self):
        self.save(b"old")
        inode = self.inode()
        someone_else = os.geteuid() + 1
        with mock.patch.object(utl_save.os, "geteuid", return_value=someone_else):
            self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt" or ROOT, "POSIX folder permissions")
    def test_writes_over_a_file_in_a_folder_that_refuses_new_files(self):
        self.save(b"old")
        inode = self.inode()
        os.chmod(self.folder, 0o500)
        try:
            self.save(b"new")
        finally:
            os.chmod(self.folder, 0o700)
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assert_nothing_left()

    def test_writes_over_a_file_it_cannot_replace(self):
        self.save(b"old")
        inode = self.inode()
        refused = PermissionError(errno.EPERM, "Operation not permitted")
        with mock.patch.object(utl_save.os, "replace", side_effect=refused):
            self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assert_nothing_left()

    def test_saves_a_file_whose_name_leaves_no_room_for_a_longer_one(self):
        name = "n" * 245 + ".dyfc"
        path = os.path.join(self.folder, name)
        self.save(b"old", path)
        self.save(b"new", path)
        self.assertEqual(self.read(path), b"new")
        self.assertEqual(os.listdir(self.folder), [name])

    def test_windows_tries_a_refused_replacement_again(self):
        self.save(b"old")
        inode = self.inode()
        replace = os.replace
        calls = []

        def busy_twice(source, target):
            calls.append(source)
            if len(calls) < 3:
                raise PermissionError(errno.EACCES, "The file is in use")
            replace(source, target)

        with mock.patch.object(utl_save, "_WINDOWS", True), mock.patch.object(
            utl_save.os, "replace", side_effect=busy_twice
        ), mock.patch.object(utl_save.time, "sleep") as sleep:
            self.save(b"new")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotEqual(self.inode(), inode)
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left()

    def test_a_failure_while_writing_over_says_the_file_may_be_incomplete(self):
        # Once the old contents are being written over: at once for contents no
        # longer than the file, after the part past its end for longer ones.
        self.save(b"old")
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))
        real_write = os.write
        for contents in (b"new", b"new and longer"):
            with self.subTest(contents=contents):
                calls = []

                def broken(fd, data, longer=len(contents) > 3):
                    calls.append(len(data))
                    if longer and len(calls) == 1:
                        return real_write(fd, data)
                    real_write(fd, b"ne")
                    raise OSError(errno.EIO, "Input/output error")

                with mock.patch.object(utl_save.os, "write", side_effect=broken):
                    with self.assertRaises(utl_save.IncompleteSaveError) as raised:
                        self.save(contents)
                self.assertEqual(raised.exception.errno, errno.EIO)
                self.assertEqual(raised.exception.strerror, "Input/output error")
                self.assert_nothing_left("other.dyfc")
                self.save(b"old")

    def test_says_the_file_may_be_incomplete_after_a_failed_replace_too(self):
        # A replace that failed in a way that may have changed the file, then a
        # write over it that failed part way, which it couldn't have begun had
        # the file been lost: the file may be incomplete, not missing.
        self.save(b"old")
        real_write = os.write
        calls = []

        def broken(fd, data):
            calls.append(len(data))
            if len(calls) == 1:
                return real_write(fd, data)
            real_write(fd, b"ne")
            raise OSError(errno.EIO, "Input/output error")

        refused = OSError(errno.EIO, "Input/output error")
        with mock.patch.object(
            utl_save.os, "replace", side_effect=refused
        ), mock.patch.object(utl_save.os, "write", side_effect=broken):
            with self.assertRaises(utl_save.IncompleteSaveError) as raised:
                self.save(b"new and longer")
        self.assertNotIsInstance(raised.exception, utl_save.MissingFileError)
        self.assertEqual(raised.exception.errno, errno.EIO)
        self.assert_nothing_left()

    def test_refuses_contents_that_cannot_fit_before_writing_over(self):
        self.save(b"old")
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))
        full = mock.Mock(total=1 << 30, free=0)
        with mock.patch.object(utl_save.shutil, "disk_usage", return_value=full):
            with self.assertRaises(OSError) as raised:
                self.save(b"longer contents")
        self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(self.read(), b"old")
        self.assert_nothing_left("other.dyfc")

    def test_gives_back_the_part_past_the_end_when_room_runs_out(self):
        # A volume can have less room than it reports free (APFS keeps some
        # back), so contents longer than the file write the part past its end
        # first, and give that back when the room runs out: the save is refused
        # with the file as it was.
        self.save(b"old contents")
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))
        real_write = os.write
        written = []

        def room_for_five(fd, data):
            room = 5 - sum(written)
            if room <= 0:
                raise OSError(errno.ENOSPC, "No space left on device")
            written.append(real_write(fd, bytes(data)[:room]))
            return written[-1]

        with mock.patch.object(utl_save.os, "write", room_for_five):
            with self.assertRaises(OSError) as raised:
                self.save(b"new contents, a good deal longer")
        self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(sum(written), 5)
        self.assertEqual(self.read(), b"old contents")
        self.assert_nothing_left("other.dyfc")
        self.save(b"new contents, a good deal longer")
        self.assertEqual(self.read(), b"new contents, a good deal longer")
        self.assert_nothing_left("other.dyfc")

    def written_over(self):
        """Patches utl_save to record the files it opens to write over, in the
        list it returns."""
        real_open = utl_save._open_to_write_over
        opened = []

        def recorded(target):
            opened.append(real_open(target))
            return opened[-1]

        patcher = mock.patch.object(utl_save, "_open_to_write_over", recorded)
        patcher.start()
        self.addCleanup(patcher.stop)
        return opened

    def test_gives_back_the_part_past_the_end_when_writing_it_out_fails(self):
        # A volume can report the room running out only as the part past the
        # end is written out to the disk, which is done before the old contents
        # are touched.
        self.save(b"old contents")
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))
        opened = self.written_over()
        real_fsync = os.fsync

        def no_room(fd):
            if fd in opened:
                raise OSError(errno.ENOSPC, "No space left on device")
            real_fsync(fd)

        with mock.patch.object(utl_save.os, "fsync", no_room):
            with self.assertRaises(OSError) as raised:
                self.save(b"new contents, a good deal longer")
        self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(self.read(), b"old contents")
        self.assert_nothing_left("other.dyfc")

    def test_says_the_file_may_be_incomplete_when_the_part_past_the_end_stays(self):
        # Should giving back the part past the end fail too, the file is left
        # longer than it was.
        self.save(b"old contents")
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))
        opened = self.written_over()
        real_write = os.write
        real_ftruncate = os.ftruncate

        def room_for_five(fd, data):
            if fd not in opened:
                return real_write(fd, data)
            if os.fstat(fd).st_size > len(b"old contents"):
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_write(fd, bytes(data)[:5])

        def refused(fd, length):
            if fd in opened:
                raise OSError(errno.EIO, "Input/output error")
            real_ftruncate(fd, length)

        with mock.patch.object(utl_save.os, "write", room_for_five), mock.patch.object(
            utl_save.os, "ftruncate", refused
        ):
            with self.assertRaises(utl_save.IncompleteSaveError) as raised:
                self.save(b"new contents, a good deal longer")
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(self.read(), b"old contents, a g")
        self.assert_nothing_left("other.dyfc")

    def test_reuses_the_room_the_old_contents_take(self):
        # Writing over the file reuses the room its contents take, so contents
        # that fit only in that and the free room are saved, and ones a byte
        # longer refused before anything is written.
        old = b"old contents"
        new = b"new contents, a good deal longer"
        self.save(old)
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))
        room = mock.Mock(total=1 << 30, free=len(new) - len(old) - 1)
        with mock.patch.object(utl_save.shutil, "disk_usage", return_value=room):
            with self.assertRaises(OSError) as raised:
                self.save(new)
            self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
            self.assertEqual(raised.exception.errno, errno.ENOSPC)
            self.assertEqual(self.read(), old)
            room.free += 1
            self.save(new)
        self.assertEqual(self.read(), new)
        self.assert_nothing_left("other.dyfc")

    def test_writes_over_a_file_on_a_volume_that_tells_no_size(self):
        # A WebDAV volume whose server keeps no quota (macOS) tells no size, and
        # shows no room free: how much there is isn't known, rather than none;
        # nor is it where the room can't be read at all.
        self.save(b"old")
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))
        untold = mock.Mock(total=0, used=0, free=0)
        unread = OSError(errno.EIO, "Input/output error")
        for case, usage in (
            ("no size", mock.Mock(return_value=untold)),
            ("unreadable", mock.Mock(side_effect=unread)),
        ):
            with self.subTest(case), mock.patch.object(
                utl_save.shutil, "disk_usage", usage
            ):
                contents = b"new contents, a good deal longer: " + case.encode()
                self.save(contents)
                self.assertEqual(self.read(), contents)
                self.assert_nothing_left("other.dyfc")

    def test_a_staged_file_cut_short_while_written_over_fails_rather_than_hang(self):
        # Someone else who can change the folder can cut the staged file short
        # just as its contents are to be written over the file: the save then
        # fails, leaving the file as it was, rather than wait forever for the
        # rest.
        self.save(b"old contents")
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))

        def cut_short(folder):
            for name in os.listdir(folder):
                if name.startswith(".dynaface-"):
                    os.truncate(os.path.join(folder, name), 2)
            return mock.Mock(total=1 << 30, free=1 << 30)

        with self.no_waiting(), mock.patch.object(
            utl_save.shutil, "disk_usage", cut_short
        ):
            with self.assertRaises(OSError) as raised:
                self.save(b"new contents, a good deal longer")
        self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
        self.assertEqual(raised.exception.errno, errno.EIO)
        self.assertEqual(self.read(), b"old contents")
        self.assert_nothing_left("other.dyfc")

    @unittest.skipIf(os.name == "nt", "POSIX links and the identity check")
    def test_a_failure_to_close_the_file_written_over_says_it_may_be_incomplete(
        self,
    ):
        # A network volume can report a write that failed only when the file is
        # closed: once the file was written over, that means it may be
        # incomplete. Nor may it take the place of an error raised already. A
        # file nothing is known of is opened when the save begins, and closed
        # at its end, the same way.
        self.save(b"old")
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))
        opened = self.written_over()
        real_close = os.close
        real_write = os.write
        another = os.path.join(self.other, "moved.dyfc")
        devnull = os.open(os.devnull, os.O_RDONLY)
        self.addCleanup(os.close, devnull)
        taken = []

        def close_fails(fd):
            real_close(fd)
            if fd in opened:
                opened.remove(fd)
                # Another thread can open a file as soon as the number is free,
                # which the save mustn't close in turn.
                os.dup2(devnull, fd)
                taken.append(fd)
                raise OSError(errno.EIO, "Input/output error")

        def no_room(fd, data):
            if fd in opened:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_write(fd, data)

        def replaced(file):
            # Someone else puts another file in the document's place.
            file.write(b"new")
            os.rename(self.path, another)
            with open(self.path, "wb") as f:
                f.write(b"another")

        def attempt(contents, writes=real_write, unknown=False, write=None):
            """The error saving contents over b"old" raises, closing the file
            written over failing."""
            self.save(b"old")
            opened.clear()
            with contextlib.ExitStack() as patches:
                patches.enter_context(
                    mock.patch.object(utl_save.os, "close", close_fails)
                )
                patches.enter_context(mock.patch.object(utl_save.os, "write", writes))
                if unknown:
                    patches.enter_context(
                        mock.patch.object(utl_save, "_stat", return_value=(True, None))
                    )
                with self.assertRaises(OSError) as raised:
                    utl_save.save_file(self.path, write or _writer(contents))
            self.assertEqual(opened, [])
            for fd in taken:
                try:
                    status = os.fstat(fd)
                except OSError:
                    self.fail("the save closed a file it had closed already")
                self.assertEqual(status.st_ino, os.fstat(devnull).st_ino)
                os.close(fd)
            taken.clear()
            return raised.exception

        for unknown in (False, True):
            with self.subTest("written over", unknown=unknown):
                error = attempt(b"new", unknown=unknown)
                self.assertIsInstance(error, utl_save.IncompleteSaveError)
                self.assertEqual(error.errno, errno.EIO)
                self.assertEqual(self.read(), b"new")
                self.assert_nothing_left("other.dyfc")
            with self.subTest("writing over failed", unknown=unknown):
                error = attempt(b"new", writes=no_room, unknown=unknown)
                self.assertIsInstance(error, utl_save.IncompleteSaveError)
                self.assertEqual(error.errno, errno.ENOSPC)
                self.assert_nothing_left("other.dyfc")
        with self.subTest("the part past the end given back"):
            error = attempt(b"new and longer", writes=no_room)
            self.assertNotIsInstance(error, utl_save.IncompleteSaveError)
            self.assertEqual(error.errno, errno.ENOSPC)
            self.assertEqual(self.read(), b"old")
            self.assert_nothing_left("other.dyfc")
        with self.subTest("another file in its place"):
            error = attempt(b"new", write=replaced)
            self.assertNotIsInstance(error, utl_save.IncompleteSaveError)
            self.assertEqual(error.errno, errno.ESTALE)
            self.assertEqual(self.read(), b"another")
            self.assertEqual(self.read(another), b"old")
            os.replace(another, self.path)
            self.assert_nothing_left("other.dyfc")

    def test_a_failure_to_close_the_staged_file_changes_nothing_the_save_did(self):
        # By then the staged file was written out, and put in the document's
        # place or written over it; or else the save failed, and that error is
        # the one to tell.
        def staged_fails_to_close(file, *args, **kwargs):
            return _FailsToClose(builtins.open(file, *args, **kwargs))

        no_room = OSError(errno.ENOSPC, "No space left on device")
        self.save(b"old")
        with mock.patch.object(utl_save, "open", staged_fails_to_close, create=True):
            with self.subTest("replaced"):
                inode = self.inode()
                with self.assertLogs("utl_save", "WARNING"):
                    self.save(b"new")
                self.assertEqual(self.read(), b"new")
                self.assertNotEqual(self.inode(), inode)
                self.assert_nothing_left()
            with self.subTest("the write failed"):
                with self.assertLogs("utl_save", "WARNING"):
                    with self.assertRaises(ValueError):
                        utl_save.save_file(self.path, _fail)
                self.assertEqual(self.read(), b"new")
                self.assert_nothing_left()
            with self.subTest("writing over failed"):
                os.link(self.path, os.path.join(self.folder, "other.dyfc"))
                with self.assertLogs("utl_save", "WARNING"), mock.patch.object(
                    utl_save.os, "write", side_effect=no_room
                ):
                    with self.assertRaises(utl_save.IncompleteSaveError) as raised:
                        self.save(b"old")
                self.assertEqual(raised.exception.errno, errno.ENOSPC)
                self.assert_nothing_left("other.dyfc")

    @unittest.skipIf(ROOT, "root can write anything")
    def test_refuses_a_read_only_file(self):
        self.save(b"old")
        os.chmod(self.path, 0o444)
        try:
            with self.assertRaises(PermissionError):
                self.save(b"new")
        finally:
            os.chmod(self.path, 0o644)
        self.assertEqual(self.read(), b"old")
        self.assert_nothing_left()

    @unittest.skipUnless(sys.platform == "darwin", "macOS extended attributes")
    def test_keeps_extended_attributes(self):
        # Finder keeps a file's tags and comments in extended attributes.
        self.save(b"old")
        xattr = "/usr/bin/xattr"
        subprocess.run(
            [xattr, "-w", "com.dynaface.test", "kept", self.path], check=True
        )
        self.save(b"new")
        value = subprocess.run(
            [xattr, "-p", "com.dynaface.test", self.path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(value.strip(), "kept")

    @unittest.skipUnless(sys.platform == "darwin", "macOS ACLs")
    def test_keeps_the_acl(self):
        import pwd

        self.save(b"old")
        entry = f"user:{pwd.getpwuid(os.geteuid()).pw_name} allow readattr"
        subprocess.run(["/bin/chmod", "+a", entry, self.path], check=True)
        self.save(b"new")
        listing = subprocess.run(
            ["/bin/ls", "-le", self.path], capture_output=True, text=True, check=True
        ).stdout
        self.assertIn(entry, listing)

    @unittest.skipUnless(sys.platform == "darwin", "macOS creation dates")
    def test_keeps_the_creation_date(self):
        self.save(b"old")
        month_ago = time.time() - 30 * 86400
        # Moves the creation date back too.
        os.utime(self.path, (month_ago, month_ago))
        created = os.stat(self.path).st_birthtime
        self.save(b"new")
        saved = os.stat(self.path)
        self.assertAlmostEqual(saved.st_birthtime, created, delta=1)
        self.assertGreater(saved.st_mtime, time.time() - 60)

    @unittest.skipUnless(MACOS, "Finder's Date Added")
    def test_keeps_the_date_added(self):
        self.save(b"old")
        month_ago = int(time.time()) - 30 * 86400
        _set_date_added(self.path, month_ago)
        inode = self.inode()
        self.save(b"new")
        self.assertNotEqual(self.inode(), inode)
        self.assertEqual(_date_added(self.path), (month_ago, 0))

    @unittest.skipUnless(MACOS, "macOS file flags")
    def test_keeps_the_hidden_and_nodump_flags(self):
        self.save(b"old")
        flags = stat.UF_HIDDEN | stat.UF_NODUMP
        os.chflags(self.path, flags)
        inode = self.inode()
        self.save(b"new")
        self.assertNotEqual(self.inode(), inode)
        self.assertEqual(os.stat(self.path).st_flags & flags, flags)
        self.assertEqual(self.read(), b"new")

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_writes_over_a_file_whose_acl_denies_removing_it(self):
        # Replacing the file would remove it, which the ACL denies; a staged
        # file given the ACL couldn't be removed either.
        self.save(b"old")
        _add_acl_entry(self.path, "everyone deny delete")
        inode = self.inode()
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assertIn("0: group:everyone deny delete", _acl(self.path))
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_writes_over_a_file_in_a_folder_that_denies_removing_files(self):
        self.save(b"old")
        _add_acl_entry(self.folder, "everyone deny delete_child")
        inode = self.inode()
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_leaves_nothing_in_a_folder_whose_new_files_cannot_be_removed(self):
        # The new document is written where it goes. It can't be removed, so
        # later saves can't replace it, and write over it from the temp folder.
        _add_acl_entry(self.folder, "everyone deny delete,file_inherit")
        for data in (b"first", b"second", b"third"):
            self.save(data)
            self.assertEqual(self.read(), data)
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_replaces_a_file_moved_into_a_folder_whose_new_files_deny_reading_them(
        self,
    ):
        # Where a network volume's server gives the staged file the folder's
        # entries, they deny its owner reading its attributes or ACL, but not
        # taking the entries off. Where the Mac would, it never has them.
        for denied in ("readattr", "readsecurity"):
            for by_server in (False, True):
                with self.subTest(denied, by_server=by_server):
                    self.save(b"old")
                    _add_acl_entry(
                        self.folder,
                        f"everyone deny {denied},write,delete,"
                        "file_inherit,only_inherit",
                    )
                    try:
                        with mock.patch.object(
                            utl_save, "_hands_down_entries", return_value=not by_server
                        ):
                            for data in (b"first", b"second"):
                                inode = self.inode()
                                self.save(data)
                                self.assertEqual(self.read(), data)
                                self.assertNotEqual(self.inode(), inode)
                        self.assertEqual(_acl(self.path), [])
                        self.assert_nothing_left()
                    finally:
                        subprocess.run(["/bin/chmod", "-N", self.folder], check=True)

    @unittest.skipUnless(MACOS, "macOS ACLs")
    def test_stages_in_the_temp_folder_where_a_staged_file_may_keep_folder_entries(
        self,
    ):
        # Entries a network volume's server hands new files could let others
        # read the staged file, so it goes in the temp folder where they can't be
        # taken off; it stays where a volume refuses an empty ACL but the file
        # has none.
        self.save(b"old")
        real = utl_save._libc()

        def refusing(reading):
            class Refusing:
                def __getattr__(self, name):
                    return getattr(real, name)

                def acl_set_fd_np(self, *args):
                    ctypes.set_errno(errno.EPERM)
                    return -1

                def acl_get_fd_np(self, *args):
                    ctypes.set_errno(reading)
                    return None

            return Refusing()

        for reading, place in ((errno.EACCES, self.temp), (errno.ENOENT, self.folder)):
            with self.subTest(errno.errorcode[reading]):
                inode = self.inode()
                staged_in = []

                def write(file):
                    staged_in.append(os.path.dirname(file.name))
                    file.write(b"new")

                with mock.patch.object(
                    utl_save, "_libc", return_value=refusing(reading)
                ), mock.patch.object(
                    utl_save, "_hands_down_entries", return_value=False
                ):
                    utl_save.save_file(self.path, write)
                self.assertEqual(staged_in, [place])
                self.assertEqual(self.read(), b"new")
                self.assertEqual(self.inode() == inode, place == self.temp)
                self.assert_nothing_left()

    def test_removes_the_empty_staged_files_earlier_sessions_left(self):
        self.save(b"old")
        left = os.path.join(self.folder, _left_name("012345"))
        open(left, "wb").close()
        # One that isn't empty may be another save's, under way, and another
        # account's is that account's to remove.
        busy = _left_name("ba9876")
        with open(os.path.join(self.folder, busy), "wb") as f:
            f.write(b"contents")
        theirs = _left_name("012345", _another_account())
        open(os.path.join(self.folder, theirs), "wb").close()
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left(busy, theirs)

    def test_stages_beside_a_file_where_another_account_left_a_staged_file(self):
        # Where each account may remove only its own files, another account's
        # staged file can't be removed, which says nothing of the user's own.
        self.save(b"old")
        theirs = os.path.join(self.folder, _left_name("012345", _another_account()))
        open(theirs, "wb").close()
        real_remove = os.remove

        def refused(path, *args):
            if path == theirs:
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return real_remove(path, *args)

        staged_in = []

        def write(file):
            staged_in.append(os.path.dirname(file.name))
            file.write(b"new")

        inode = self.inode()
        with mock.patch.object(utl_save.os, "remove", side_effect=refused):
            utl_save.save_file(self.path, write)
        self.assertEqual(staged_in, [self.folder])
        self.assertEqual(self.read(), b"new")
        self.assertNotEqual(self.inode(), inode)
        self.assert_nothing_left(os.path.basename(theirs))

    @unittest.skipUnless(hasattr(os, "geteuid"), "POSIX owners")
    def test_keeps_a_staged_file_another_owner_left(self):
        # A name can be copied; where files have owners, the owner decides.
        self.save(b"old")
        left = _left_name("012345")
        open(os.path.join(self.folder, left), "wb").close()
        real_lstat = os.lstat

        def another_owner(path, *args, **kwargs):
            status = real_lstat(path, *args, **kwargs)
            if os.path.basename(path) != left:
                return status
            fields = list(status)
            fields[stat.ST_UID] = os.geteuid() + 1
            return os.stat_result(fields)

        with mock.patch.object(utl_save.os, "lstat", another_owner):
            self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left(left)

    def test_stages_in_the_temp_folder_where_an_earlier_session_left_a_staged_file(
        self,
    ):
        # The empty staged file left in a folder whose new files can't be
        # removed records that from one session to the next.
        self.save(b"old")
        left = os.path.join(self.folder, _left_name("012345"))
        open(left, "wb").close()
        real_remove = os.remove

        def refused(path, *args):
            if path == left:
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return real_remove(path, *args)

        staged_in = []

        def write(file):
            staged_in.append(os.path.dirname(file.name))
            file.write(b"new")

        with mock.patch.object(utl_save.os, "remove", side_effect=refused):
            utl_save.save_file(self.path, write)
        self.assertEqual(staged_in, [self.temp])
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left(os.path.basename(left))

    def test_a_staged_file_left_in_one_session_keeps_later_ones_from_leaving_more(
        self,
    ):
        # In a folder whose new files can't be removed, a save that can't
        # replace the document leaves its staged file, and the next session's
        # first save finds it there, by its name, and stages elsewhere: on a Mac
        # too, whose host name changes with the network from one session to the
        # next.
        self.save(b"old")
        self.addCleanup(utl_save._account.cache_clear)
        real_remove = os.remove

        def refused(path, *args):
            name = os.path.basename(path)
            if os.path.dirname(path) == self.folder and name.startswith(".dynaface-"):
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return real_remove(path, *args)

        replace = PermissionError(errno.EPERM, "Operation not permitted")
        with mock.patch.object(
            utl_save.os, "remove", side_effect=refused
        ), mock.patch.object(utl_save.os, "replace", side_effect=replace):
            for session in range(3):
                utl_save._STUCK_FOLDERS.clear()
                utl_save._CHECKED_FOLDERS.clear()
                utl_save._account.cache_clear()
                network = (
                    mock.patch.object(
                        utl_save.socket,
                        "gethostname",
                        return_value=f"dhcp-10-0-0-{session}.example.com",
                    )
                    if MACOS
                    else contextlib.nullcontext()
                )
                logged = (
                    self.assertLogs("utl_save", "WARNING")
                    if session == 0
                    else contextlib.nullcontext()
                )
                with network, logged:
                    self.save(b"new %d" % session)
                self.assertEqual(self.read(), b"new %d" % session)
                left = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
                self.assertEqual(len(left), 1)
                self.assertEqual(os.listdir(self.temp), [])

    def test_tells_accounts_apart_by_user_and_computer(self):
        # Accounts of one name on two computers can save to one shared folder,
        # and each knows its own staged files by its tag.
        self.addCleanup(utl_save._account.cache_clear)
        names = []
        for computer, user in (
            ("front-desk", "reception"),
            ("front-desk", "doctor"),
            ("exam-room", "reception"),
        ):
            with mock.patch.object(
                utl_save, "_computer", return_value=computer
            ), mock.patch.object(utl_save.getpass, "getuser", return_value=user):
                utl_save._account.cache_clear()
                name = utl_save._staged_name()
                self.assertEqual(len(name), 26)
                self.assertTrue(utl_save._is_own_staged(name))
                self.assertFalse(any(utl_save._is_own_staged(n) for n in names))
                names.append(name)

    def test_knows_the_computer_by_what_the_network_cannot_change(self):
        # A Mac's host name, unless the user set one, is whatever the network it
        # is on calls its address, so a Mac is known by its hardware UUID;
        # elsewhere the host name is the computer's own name.
        network_names = ["front-desk.local", "dhcp-10-0-0-7.example.com"]
        computers = []
        for network_name in network_names:
            with mock.patch.object(
                utl_save.socket, "gethostname", return_value=network_name
            ):
                computers.append(utl_save._computer())
        if MACOS:
            self.assertEqual(computers[0], computers[1])
            self.assertRegex(computers[0], r"\A[0-9a-f]{32}\Z")
            self.assertNotEqual(computers[0], "0" * 32)
        else:
            self.assertEqual(computers, network_names)

    def test_knows_the_computer_by_its_host_name_where_nothing_else_tells_it(self):
        # Off macOS, and on a Mac that tells no UUID, whatever the call left in
        # its buffer, or one of zeros, which tells it from no other; and a
        # computer whose host name can't be read either is known by nothing,
        # rather than fail the save.
        waits = []

        def refused(uuid, wait):
            waits.append((wait._obj.tv_sec, wait._obj.tv_nsec))
            ctypes.memmove(uuid, b"\x01" * 16, 16)
            return -1

        unreadable = OSError(errno.EIO, "Input/output error")
        no_uuid = mock.Mock(**{"gethostuuid.side_effect": refused})
        zeros = mock.Mock(**{"gethostuuid.return_value": 0})
        no_calls = OSError(errno.ENOSYS, "Function not implemented")
        for case, patcher in (
            (
                "off macOS",
                mock.patch.object(utl_save, "sys", _Module(sys, platform="win32")),
            ),
            ("no UUID", mock.patch.object(utl_save, "_libc", return_value=no_uuid)),
            ("zeros", mock.patch.object(utl_save, "_libc", return_value=zeros)),
            ("no C calls", mock.patch.object(utl_save, "_libc", side_effect=no_calls)),
        ):
            with self.subTest(case), patcher:
                with mock.patch.object(
                    utl_save.socket, "gethostname", return_value="exam-room"
                ):
                    self.assertEqual(utl_save._computer(), "exam-room")
                with mock.patch.object(
                    utl_save.socket, "gethostname", side_effect=unreadable
                ):
                    self.assertEqual(utl_save._computer(), "")
        if MACOS:
            # A wait of none would be one without end.
            self.assertTrue(waits)
            self.assertNotIn((0, 0), waits)

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_never_touches_a_file_linked_in_the_staged_file_s_place(self):
        # Someone else who can change a shared folder can rename the staged file
        # while the document is written, and put a link to a private file of the
        # user's in its place. The save must neither read, change nor move that
        # file: it writes the document over instead. Its own staged file stays
        # where it was moved, with no more than the document's permissions. A
        # group of the document's own is given to the staged file too, so that
        # too goes through the open file.
        self.save(b"old")
        os.chmod(self.path, 0o644)
        self.give_another_group(self.path)
        if MACOS:
            _add_acl_entry(self.path, "group:staff allow read")
            os.chflags(self.path, stat.UF_NODUMP)
        for link in (os.symlink, os.link):
            with self.subTest(link.__name__):
                private = self.private_file()
                moved = os.path.join(self.other, "moved")
                names = []

                def write(file):
                    file.write(b"new")
                    names.append(file.name)
                    os.rename(file.name, moved)
                    link(private, file.name)

                utl_save.save_file(self.path, write)
                self.assertEqual(self.read(), b"new")
                self.assertFalse(os.path.islink(self.path))
                self.assert_untouched(private)
                self.assertEqual(os.stat(moved).st_mode & 0o777, 0o644)
                for path in (names[0], moved, private):
                    os.remove(path)
                self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_writes_over_rather_than_place_a_staged_file_with_another_link(self):
        # A link to the staged file made elsewhere would reach the document too.
        self.save(b"old")
        inode = self.inode()
        extra = os.path.join(self.other, "extra")

        def write(file):
            file.write(b"new")
            os.link(file.name, extra)

        utl_save.save_file(self.path, write)
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assertEqual(os.path.getsize(extra), 0)
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_writes_over_from_its_own_staged_file_once_another_has_its_name(self):
        self.save(b"old")
        other = os.path.join(self.folder, "other.dyfc")
        os.link(self.path, other)
        private = self.private_file()
        moved = os.path.join(self.other, "moved")
        names = []

        def write(file):
            file.write(b"new")
            names.append(file.name)
            os.rename(file.name, moved)
            os.symlink(private, file.name)

        utl_save.save_file(self.path, write)
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.read(other), b"new")
        self.assert_untouched(private)
        # Still private: a file with other hard links is written over before
        # the staged file is given its permissions.
        self.assertEqual(os.stat(moved).st_mode & 0o777, 0o600)
        os.remove(names[0])
        self.assert_nothing_left("other.dyfc")

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_a_link_put_in_the_staged_file_s_place_at_the_last_moment_changes_no_file(
        self,
    ):
        # A link put in the staged file's place after the save last looked at it
        # takes the document's place: what anyone who can change the folder can
        # do anyway. The file it links to keeps its permissions, as all the save
        # gave the staged file went through the open file.
        self.save(b"old")
        os.chmod(self.path, 0o644)
        if MACOS:
            _add_acl_entry(self.path, "group:staff allow read")
        private = self.private_file()
        moved = os.path.join(self.other, "moved")
        real_replace = os.replace

        def swap_then_replace(source, target):
            os.rename(source, moved)
            os.symlink(private, source)
            real_replace(source, target)

        with mock.patch.object(utl_save.os, "replace", swap_then_replace):
            self.save(b"new")
        self.assertEqual(os.readlink(self.path), private)
        self.assert_untouched(private)

    @unittest.skipUnless(MACOS, "the replace reads the document's metadata on macOS")
    def test_refuses_to_save_into_a_file_linked_in_the_document_s_place(self):
        # The save reads the document's metadata, and writes over it, only while
        # it is the file the save began with.
        cases = (
            ("symbolic link", os.symlink, False),
            ("hard link", os.link, False),
            ("hard link, writing over", os.link, True),
        )
        for name, link, hard_linked in cases:
            with self.subTest(name):
                self.save(b"old")
                other = os.path.join(self.folder, "other.dyfc")
                if hard_linked:
                    os.link(self.path, other)
                private = self.private_file()
                moved = os.path.join(self.other, "moved.dyfc")

                def write(file):
                    file.write(b"new")
                    os.rename(self.path, moved)
                    link(private, self.path)

                with self.assertRaises(OSError):
                    utl_save.save_file(self.path, write)
                self.assert_untouched(private)
                for path in (self.path, moved, private):
                    os.remove(path)
                if hard_linked:
                    os.remove(other)
                self.assertEqual(os.listdir(self.folder), [])
                self.assertEqual(os.listdir(self.temp), [])

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_leaves_its_staged_file_alone_once_it_is_in_the_document_s_place(self):
        # Someone else who can change the folder can move the staged file into
        # the document's place themselves. The save, unable to tell what it would
        # write over, stops, but mustn't empty what is now the document.
        self.save(b"old")

        def write(file):
            file.write(b"new")
            file.flush()
            os.rename(file.name, self.path)

        with self.assertRaises(OSError):
            utl_save.save_file(self.path, write)
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_keeps_a_replace_reported_failing_that_was_done(self):
        # A network volume can report a rename failing that it did; reported
        # busy, the staged file is closed, as where a volume renames no file
        # held open, and still known in the document's place.
        real_replace = os.replace
        for code in (errno.ENOENT, errno.EBUSY):
            with self.subTest(errno=errno.errorcode[code]):
                self.save(b"old")
                inode = self.inode()

                def done_but_failed(source, target):
                    real_replace(source, target)
                    raise OSError(code, os.strerror(code), source)

                with mock.patch.object(utl_save.os, "replace", done_but_failed):
                    self.save(b"new")
                self.assertEqual(self.read(), b"new")
                self.assertNotEqual(self.inode(), inode)
                self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_tries_again_a_replace_that_removed_the_file_and_failed(self):
        # A network volume can remove the file and then fail to rename the
        # staged file into its place: smbfs does it in those two steps, and a
        # WebDAV server can fail between them. Once the file is gone, trying
        # again has nothing to remove.
        real_replace = os.replace

        def removed_then_failed(code):
            calls = []

            def replace(source, target):
                calls.append(source)
                if len(calls) == 1:
                    os.remove(target)
                    raise OSError(code, os.strerror(code), source)
                real_replace(source, target)

            return replace

        for case in ("smbfs", "WebDAV"):
            with self.subTest(case):
                self.save(b"old")
                inode = self.inode()
                if case == "smbfs":
                    volume = mock.patch.object(
                        utl_save.os, "replace", removed_then_failed(errno.ETIMEDOUT)
                    )
                else:
                    volume = self.renames_no_open_file(
                        replace=removed_then_failed(errno.ENOENT)
                    )
                with volume, self.assertNoLogs("utl_save", "WARNING"):
                    self.save(b"new")
                self.assertEqual(self.read(), b"new")
                self.assertNotEqual(self.inode(), inode)
                self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_keeps_its_staged_file_open_when_a_replace_fails_but_for_being_busy(self):
        # Only a volume that renames no file held open, reporting it busy, has
        # the staged file closed; any other failure leaves it open while the
        # replace is tried again, so no one can put another file in its place
        # meanwhile.
        self.save(b"old")
        real_replace = os.replace
        real_release = utl_save._NewFile.release
        calls, released = [], []

        def refused_once(source, target):
            calls.append(source)
            if len(calls) == 1:
                raise PermissionError(errno.EACCES, "Permission denied", source)
            real_replace(source, target)

        def recorded(staged):
            released.append(staged.path)
            real_release(staged)

        with mock.patch.object(utl_save.os, "replace", refused_once), mock.patch.object(
            utl_save._NewFile, "release", recorded
        ):
            self.save(b"new")
        self.assertEqual(len(calls), 2)
        self.assertEqual(released, [])
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_never_empties_a_closed_file_by_name_where_files_have_no_number(self):
        # With no number to know it by, a file of the same size at its name
        # could be another's, so it is never emptied by name.
        path = os.path.join(self.folder, utl_save._staged_name())
        staged = utl_save._NewFile(path, open(path, "x+b"))
        staged.file.write(b"new")
        staged.file.flush()
        real_fstat = os.fstat
        with mock.patch.object(
            utl_save.os, "fstat", lambda fd: _Unnumbered(real_fstat(fd))
        ):
            staged.release()
            self.assertFalse(staged.empty(by_name=True))
        self.assertEqual(self.read(path), b"new")
        os.remove(path)

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_keeps_the_new_contents_when_a_replace_removed_the_file_and_failed(self):
        # Should trying again fail too, the file is gone: the staged file, with
        # the new contents, is kept rather than removed, and the save says the
        # file may be missing rather than as it was.
        real_remove = os.remove

        def removed_then_failed(source, target):
            with contextlib.suppress(FileNotFoundError):
                real_remove(target)
            raise OSError(errno.EIO, "Input/output error", source)

        for case in ("smbfs", "WebDAV"):
            with self.subTest(case):
                self.save(b"old")
                if case == "smbfs":
                    volume = mock.patch.object(
                        utl_save.os, "replace", removed_then_failed
                    )
                else:
                    volume = self.renames_no_open_file(replace=removed_then_failed)
                with volume, self.assertNoLogs("utl_save", "WARNING"):
                    with self.assertRaises(utl_save.MissingFileError) as raised:
                        self.save(b"new")
                # Raised as one that may have left the file incomplete, too.
                self.assertIsInstance(raised.exception, utl_save.IncompleteSaveError)
                self.assertEqual(raised.exception.errno, errno.ENOENT)
                kept = raised.exception.kept
                self.assertEqual(os.path.dirname(kept), self.folder)
                self.assertEqual(self.read(kept), b"new")
                self.assertFalse(os.path.lexists(self.path))
                self.assertEqual(utl_save._STUCK_FOLDERS, set())
                os.remove(kept)
                self.assertEqual(os.listdir(self.folder), [])
                self.assertEqual(os.listdir(self.temp), [])

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_keeps_the_new_contents_when_the_file_cannot_be_found_after_a_replace(
        self,
    ):
        # smbfs removes the file first, which another computer holding it open
        # only marks for removal: renaming the staged file fails then, and the
        # file can no longer be opened or looked at, nor is it there once the
        # other computer closes it.
        self.save(b"old")
        marked = []

        def removal_pending(source, target):
            marked.append(target)
            raise FileExistsError(errno.EEXIST, "File exists", target)

        def refused_once_marked(real):
            def call(path, *args, **kwargs):
                if path in marked:
                    raise PermissionError(errno.EACCES, "Permission denied", path)
                return real(path, *args, **kwargs)

            return call

        with mock.patch.object(
            utl_save.os, "replace", removal_pending
        ), mock.patch.object(
            utl_save.os, "lstat", refused_once_marked(os.lstat)
        ), mock.patch.object(
            utl_save.os, "open", refused_once_marked(os.open)
        ):
            with self.assertRaises(utl_save.MissingFileError) as raised:
                self.save(b"new")
        self.assertEqual(raised.exception.errno, errno.EACCES)
        self.assertEqual(self.read(raised.exception.kept), b"new")
        os.remove(raised.exception.kept)
        self.assertEqual(self.read(), b"old")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_keeps_the_new_contents_when_the_file_lost_its_own_after_a_replace(self):
        # A WebDAV server that removed the file and failed can show it as an
        # empty one, which the volume still numbers as the file.
        self.save(b"old")
        full = mock.Mock(total=1 << 30, free=0)

        def emptied_then_failed(source, target):
            os.truncate(target, 0)
            raise OSError(errno.EIO, "Input/output error", source)

        with mock.patch.object(
            utl_save.os, "replace", emptied_then_failed
        ), mock.patch.object(utl_save.shutil, "disk_usage", return_value=full):
            with self.assertRaises(utl_save.MissingFileError) as raised:
                self.save(b"longer contents")
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(self.read(raised.exception.kept), b"longer contents")
        os.remove(raised.exception.kept)
        self.assertEqual(self.read(), b"")
        self.assert_nothing_left()

    def test_removes_the_staged_file_when_a_failed_replace_left_the_file_as_it_was(
        self,
    ):
        # A replace refused before it changed anything, and a write over the
        # file that failed too, leave nothing behind, and the file as it was.
        self.save(b"old")
        refused = PermissionError(errno.EPERM, "Operation not permitted")
        full = mock.Mock(total=1 << 30, free=0)
        with mock.patch.object(
            utl_save.os, "replace", side_effect=refused
        ), mock.patch.object(utl_save.shutil, "disk_usage", return_value=full):
            with self.assertRaises(OSError) as raised:
                self.save(b"longer contents")
        self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(self.read(), b"old")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_keeps_the_new_contents_when_another_file_took_the_file_s_place(self):
        # A WebDAV server (Apache) that removed the file and failed puts an
        # empty file in its place once writing over it locks it, which the
        # volume numbers anew. The save can't write over that, and keeps the new
        # contents rather than leave them nowhere.
        self.save(b"old")
        empty = os.path.join(self.other, "empty")
        aside = os.path.join(self.other, "old")

        def replaced_then_failed(source, target):
            if not os.path.lexists(aside):
                # Moved away rather than removed, so that its number stays in
                # use: ext4 can give a removed file's number to the next one.
                os.rename(target, aside)
                with open(empty, "wb"):
                    pass
                os.rename(empty, target)
            raise OSError(errno.EIO, "Input/output error", source)

        with mock.patch.object(utl_save.os, "replace", replaced_then_failed):
            with self.assertRaises(utl_save.MissingFileError) as raised:
                self.save(b"new")
        self.assertEqual(raised.exception.errno, errno.ESTALE)
        self.assertEqual(self.read(raised.exception.kept), b"new")
        os.remove(raised.exception.kept)
        self.assertEqual(self.read(), b"")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_says_the_file_was_not_changed_while_another_holds_it(self):
        # Another computer writing the document on a WebDAV volume holds its
        # lock, so the server refuses the replace and writing over it alike as
        # busy, before it changes anything. The volume numbers a file by its
        # name, so the other computer's version shows as the same file, of
        # another size, yet the save says the file wasn't changed, and leaves
        # nothing.
        self.save(b"old")
        real_open = os.open

        def locked(source, target):
            if self.read(target) == b"old":
                with open(target, "ab") as f:
                    f.write(b" and theirs")
            raise OSError(errno.EBUSY, "Resource busy", source)

        def refused_for_writing(path, flags, *args, **kwargs):
            if path == self.path and flags & (os.O_WRONLY | os.O_RDWR):
                raise OSError(errno.EBUSY, "Resource busy", path)
            return real_open(path, flags, *args, **kwargs)

        with self.renames_no_open_file(replace=locked), mock.patch.object(
            utl_save.os, "open", refused_for_writing
        ), self.assertNoLogs("utl_save", "WARNING"):
            with self.assertRaises(OSError) as raised:
                self.save(b"new")
        self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
        self.assertEqual(raised.exception.errno, errno.EBUSY)
        self.assertEqual(self.read(), b"old and theirs")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_keeps_the_new_contents_when_a_replace_removed_the_file_and_was_refused_busy(
        self,
    ):
        # A busy refusal can come once the file is gone: smbfs removes it, then
        # the server refuses to rename the staged file while someone else holds
        # it (a scanner, say), and SabreDAV removes it before refusing to move
        # a file it holds locked, where Apache would put an empty file in its
        # place once writing over it locks it. The new contents are kept.
        real_open = os.open
        removed = os.path.join(self.other, "removed")

        def removed_then_refused(source, target):
            if os.path.lexists(target):
                # Moved away rather than removed, so that its number stays in
                # use: ext4 can give a removed file's number to the next one.
                os.rename(target, removed)
            raise OSError(errno.EBUSY, "Resource busy", source)

        def locked_empty(path, flags, *args, **kwargs):
            if path == self.path and flags & (os.O_WRONLY | os.O_RDWR):
                if not os.path.lexists(path):
                    with open(path, "wb"):
                        pass
                raise OSError(errno.EBUSY, "Resource busy", path)
            return real_open(path, flags, *args, **kwargs)

        for case in ("smbfs", "SabreDAV"):
            with self.subTest(case):
                self.save(b"old")
                volume = contextlib.ExitStack()
                if case == "smbfs":
                    volume.enter_context(
                        mock.patch.object(utl_save.os, "replace", removed_then_refused)
                    )
                else:
                    volume.enter_context(
                        self.renames_no_open_file(replace=removed_then_refused)
                    )
                    volume.enter_context(
                        mock.patch.object(utl_save.os, "open", locked_empty)
                    )
                with volume, self.assertNoLogs("utl_save", "WARNING"):
                    with self.assertRaises(utl_save.MissingFileError) as raised:
                        self.save(b"new")
                kept = raised.exception.kept
                self.assertEqual(os.path.dirname(kept), self.folder)
                self.assertEqual(self.read(kept), b"new")
                os.remove(kept)
                self.assertEqual(self.read(removed), b"old")
                os.remove(removed)
                if case == "smbfs":
                    self.assertEqual(raised.exception.errno, errno.ENOENT)
                    self.assertEqual(os.listdir(self.folder), [])
                else:
                    self.assertEqual(raised.exception.errno, errno.EBUSY)
                    self.assertEqual(self.read(), b"")
                    self.assert_nothing_left()
                    os.remove(self.path)
                self.assertEqual(os.listdir(self.temp), [])

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_keeps_the_new_contents_when_a_replace_refused_busy_can_t_be_tried_again(
        self,
    ):
        # smbfs removed the file, then refused to rename the staged file someone
        # else holds, and once the save closed the staged file it can't open it
        # by name again (the holder took it alone, say), to try the replace
        # again or to write the file over. The one refusal, made while the save
        # held the staged file open, tells that the file may be gone, and the
        # new contents are kept.
        self.save(b"old")
        real_open = os.open
        tried = []

        def removed_then_refused(source, target):
            tried.append(source)
            if os.path.lexists(target):
                os.remove(target)
            raise OSError(errno.EBUSY, "Resource busy", source)

        def held_alone(path, flags, *args, **kwargs):
            if tried and os.path.basename(path).startswith(".dynaface-"):
                raise OSError(errno.EBUSY, "Resource busy", path)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            utl_save.os, "replace", removed_then_refused
        ), mock.patch.object(utl_save.os, "open", held_alone), self.assertNoLogs(
            "utl_save", "WARNING"
        ):
            with self.assertRaises(utl_save.MissingFileError) as raised:
                self.save(b"new")
        self.assertEqual(len(tried), 1)
        self.assertEqual(raised.exception.errno, errno.EBUSY)
        kept = raised.exception.kept
        self.assertEqual(os.path.dirname(kept), self.folder)
        self.assertEqual(self.read(kept), b"new")
        os.remove(kept)
        self.assertFalse(os.path.lexists(self.path))
        self.assertEqual(os.listdir(self.folder), [])
        self.assertEqual(os.listdir(self.temp), [])

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_says_the_file_was_not_changed_when_the_volume_numbers_it_anew(self):
        # A WebDAV volume (macOS) gives an untouched file a new number when the
        # server fails a request to lock or read it, as writing over it after a
        # failed replace does. That isn't taken for another file: the save
        # can't write over it, and says it wasn't changed, leaving nothing,
        # whether the replaces were refused busy or failed otherwise.
        anew = os.path.join(self.other, "anew")

        def failed_with(code):
            calls = []

            def replace(source, target):
                calls.append(source)
                if len(calls) == 1:
                    # The same contents, in a file the old one's number can't
                    # go to, as both are in use.
                    with open(anew, "wb") as f:
                        f.write(self.read(target))
                    os.rename(anew, target)
                raise OSError(code, os.strerror(code), source)

            return replace

        for code in (errno.EBUSY, errno.ENOENT):
            with self.subTest(errno.errorcode[code]):
                self.save(b"old")
                inode = self.inode()
                with mock.patch.object(
                    utl_save.os, "replace", failed_with(code)
                ), self.assertNoLogs("utl_save", "WARNING"):
                    with self.assertRaises(OSError) as raised:
                        self.save(b"new")
                self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
                self.assertEqual(raised.exception.errno, errno.ESTALE)
                self.assertEqual(self.read(), b"old")
                self.assertNotEqual(self.inode(), inode)
                self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_keeps_the_new_contents_once_a_replace_failed_otherwise_than_busy(self):
        # A replace that timed out may have changed the file, here emptied it,
        # as a WebDAV server can, while the volume still numbers it as the
        # file. Trying again, refused busy (someone else holds the staged file
        # now), doesn't make that a refusal that changed nothing.
        self.save(b"old")
        real_open = os.open
        calls = []

        def timed_out_then_refused(source, target):
            calls.append(source)
            if len(calls) == 1:
                os.truncate(target, 0)
                raise TimeoutError(errno.ETIMEDOUT, "Operation timed out", source)
            raise OSError(errno.EBUSY, "Resource busy", source)

        def refused_for_writing(path, flags, *args, **kwargs):
            if path == self.path and flags & (os.O_WRONLY | os.O_RDWR):
                raise OSError(errno.EBUSY, "Resource busy", path)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(
            utl_save.os, "replace", timed_out_then_refused
        ), mock.patch.object(
            utl_save.os, "open", refused_for_writing
        ), self.assertNoLogs(
            "utl_save", "WARNING"
        ):
            with self.assertRaises(utl_save.MissingFileError) as raised:
                self.save(b"new")
        self.assertEqual(len(calls), 3)
        self.assertEqual(raised.exception.errno, errno.EBUSY)
        self.assertEqual(self.read(raised.exception.kept), b"new")
        os.remove(raised.exception.kept)
        self.assertEqual(self.read(), b"")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_keeps_the_new_contents_when_a_link_took_the_file_s_place(self):
        # A link put in the file's place once a replace failed, or was refused
        # busy, is neither written through nor taken for the file, even one of
        # its size.
        aside = os.path.join(self.other, "old")

        def linked_then_failed(code):
            def replace(source, target):
                if not os.path.lexists(aside):
                    os.rename(target, aside)
                    os.symlink("abc", target)
                raise OSError(code, os.strerror(code), source)

            return replace

        for code in (errno.EIO, errno.EBUSY):
            with self.subTest(errno.errorcode[code]):
                self.save(b"old")
                with mock.patch.object(
                    utl_save.os, "replace", linked_then_failed(code)
                ):
                    with self.assertRaises(utl_save.MissingFileError) as raised:
                        self.save(b"new")
                self.assertEqual(os.lstat(self.path).st_size, os.lstat(aside).st_size)
                self.assertEqual(raised.exception.errno, errno.ELOOP)
                self.assertEqual(self.read(raised.exception.kept), b"new")
                os.remove(raised.exception.kept)
                self.assertEqual(os.readlink(self.path), "abc")
                self.assertEqual(self.read(aside), b"old")
                os.remove(aside)
                self.assert_nothing_left()
                # A link to nothing, which tearDown() can't clear ACLs through.
                os.remove(self.path)

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_says_the_file_was_not_changed_when_the_volume_stops_after_busy_refusals(
        self,
    ):
        # A replace refused busy, as a WebDAV lock refuses one, is never one
        # done, so a staged file the volume no longer shows, once it stops
        # answering, isn't taken for one in the file's place: the save says the
        # file wasn't changed, and the staged file stays where the volume took
        # it.
        self.save(b"old")
        stopped = []

        def refused_then_stopped(source, target):
            stopped.append(True)
            raise OSError(errno.EBUSY, "Resource busy", source)

        def unless_stopped(real):
            def call(path, *args, **kwargs):
                if stopped and os.path.dirname(path) == self.folder:
                    raise OSError(errno.ENXIO, "Device not configured", path)
                return real(path, *args, **kwargs)

            return call

        with self.renames_no_open_file(replace=refused_then_stopped), mock.patch.object(
            utl_save.os, "open", unless_stopped(os.open)
        ), mock.patch.object(
            utl_save.os, "lstat", unless_stopped(os.lstat)
        ), self.assertNoLogs(
            "utl_save", "WARNING"
        ):
            with self.assertRaises(OSError) as raised:
                self.save(b"new")
        self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
        self.assertEqual(raised.exception.errno, errno.ENXIO)
        self.assertEqual(len(stopped), 1)
        self.assertEqual(self.read(), b"old")
        (left,) = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
        self.assertEqual(self.read(os.path.join(self.folder, left)), b"new")
        os.remove(os.path.join(self.folder, left))
        self.assertEqual(utl_save._STUCK_FOLDERS, set())
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_says_the_file_was_not_changed_when_the_volume_fails_before_any_replace(
        self,
    ):
        # A volume that stops answering once the contents are staged fails the
        # write over the file, and any look at it, but a save that never tried
        # to replace the file can't have changed it: a document written over
        # (another hard link), one staged in the temp folder, and one whose
        # metadata couldn't be given to the staged file.
        dropped = []

        def write(file):
            file.write(b"new")
            dropped.append(True)

        def unless_dropped(real):
            def call(path, *args, **kwargs):
                if dropped and path == self.path:
                    raise TimeoutError(errno.ETIMEDOUT, "Operation timed out", path)
                return real(path, *args, **kwargs)

            return call

        unanswered = TimeoutError(errno.ETIMEDOUT, "Operation timed out")
        for case in ("other hard links", "staged in the temp folder", "metadata"):
            with self.subTest(case):
                self.save(b"old")
                others = ()
                patches = contextlib.ExitStack()
                if case == "other hard links":
                    os.link(self.path, os.path.join(self.folder, "other.dyfc"))
                    others = ("other.dyfc",)
                elif case == "staged in the temp folder":
                    utl_save._STUCK_FOLDERS.add(self.folder)
                else:
                    patches.enter_context(
                        mock.patch.object(
                            utl_save, "_take_metadata", side_effect=unanswered
                        )
                    )
                dropped.clear()
                with patches, mock.patch.object(
                    utl_save.os, "open", unless_dropped(os.open)
                ), mock.patch.object(utl_save.os, "lstat", unless_dropped(os.lstat)):
                    with self.assertRaises(OSError) as raised:
                        utl_save.save_file(self.path, write)
                self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
                self.assertEqual(raised.exception.errno, errno.ETIMEDOUT)
                self.assertEqual(self.read(), b"old")
                self.assert_nothing_left(*others)
                for name in others:
                    os.remove(os.path.join(self.folder, name))
                utl_save._STUCK_FOLDERS.clear()

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_says_the_file_may_hold_the_new_contents_when_a_replace_lost_its_answer(
        self,
    ):
        # A WebDAV server can carry out a rename whose answer is lost, and asked
        # again, no longer have the staged file. The volume (macOS) still shows
        # the document as it was, from its cache, for a while, so the rename
        # can't be told done: the save says the file may hold the new contents,
        # keeping nothing, as the staged file is the document now.
        self.save(b"old")
        cached = os.lstat(self.path)
        real_replace = os.replace

        def done_but_lost(source, target):
            real_replace(source, target)
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", source)

        def from_cache(real):
            def status(path, *args, **kwargs):
                if path == self.path:
                    return cached
                return real(path, *args, **kwargs)

            return status

        with self.renames_no_open_file(replace=done_but_lost), mock.patch.object(
            utl_save.os, "lstat", from_cache(os.lstat)
        ), mock.patch.object(
            utl_save, "_status_through_file", from_cache(utl_save._status_through_file)
        ), self.assertNoLogs(
            "utl_save", "WARNING"
        ):
            with self.assertRaises(utl_save.MissingFileError) as raised:
                self.save(b"new contents")
        self.assertEqual(raised.exception.errno, errno.ENOENT)
        self.assertIsNone(raised.exception.kept)
        self.assertEqual(self.read(), b"new contents")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_an_interrupt_while_writing_over_is_raised_as_it_came(self):
        # Even once a replace removed the file: it isn't taken for a failure.
        self.save(b"old")
        real_remove = os.remove

        def removed_then_failed(source, target):
            with contextlib.suppress(FileNotFoundError):
                real_remove(target)
            raise OSError(errno.EIO, "Input/output error", source)

        with mock.patch.object(
            utl_save.os, "replace", removed_then_failed
        ), mock.patch.object(utl_save, "_write_over", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.save(b"new")
        for name in os.listdir(self.folder):
            os.remove(os.path.join(self.folder, name))

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_saves_on_a_volume_that_renames_no_file_held_open(self):
        # A WebDAV volume (macOS) refuses as busy to rename or remove a file
        # that is open. The staged file is closed then and renamed by name, and
        # a file the save made and must remove is closed to be removed: nothing
        # is left behind, nor is the folder given up on.
        self.save(b"old")
        os.chmod(self.path, 0o640)
        other = os.path.join(self.folder, "other.dyfc")
        with self.renames_no_open_file():
            with self.subTest("replaced"), self.assertNoLogs("utl_save", "WARNING"):
                inode = self.inode()
                self.save(b"new")
                self.assertEqual(self.read(), b"new")
                self.assertNotEqual(self.inode(), inode)
                self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o640)
                self.assert_nothing_left()
            with self.subTest("written over"), self.assertNoLogs("utl_save", "WARNING"):
                os.link(self.path, other)
                inode = self.inode()
                self.save(b"newer")
                self.assertEqual(self.read(other), b"newer")
                self.assertEqual(self.inode(), inode)
                self.assert_nothing_left("other.dyfc")
            with self.subTest("a new file that failed"), self.assertNoLogs(
                "utl_save", "WARNING"
            ):
                with self.assertRaises(ValueError):
                    utl_save.save_file(os.path.join(self.folder, "new.dyfc"), _fail)
                self.assert_nothing_left("other.dyfc")
        self.assertEqual(utl_save._STUCK_FOLDERS, set())

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_writes_over_from_its_closed_staged_file_when_renaming_it_fails(self):
        # Once closed to be renamed, the staged file is opened again by name,
        # while it is still the same file, should the rename fail: for reading
        # alone, as a WebDAV volume locks a file opened for writing (see
        # test_opens_its_closed_staged_file_to_write_only_where_it_is_known);
        # and only the opening waits on nothing.
        self.save(b"old")
        inode = self.inode()
        real_reopen = utl_save._reopen
        opened = []

        def refused(source, target):
            raise PermissionError(errno.EACCES, "Permission denied", source)

        def reopened(path, flags):
            fd = real_reopen(path, flags)
            opened.append((flags & os.O_ACCMODE, os.get_blocking(fd)))
            return fd

        with self.renames_no_open_file(replace=refused), mock.patch.object(
            utl_save, "_reopen", reopened
        ), self.assertNoLogs("utl_save", "WARNING"):
            self.save(b"new")
        self.assertEqual(opened, [(os.O_RDONLY, True)])
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_never_writes_over_from_its_closed_staged_file_once_its_size_changed(
        self,
    ):
        # Nothing the save does changes the staged file's size once it is
        # closed. A WebDAV server can make an empty file of one it no longer
        # has, which the volume still numbers as the staged file.
        self.save(b"old")
        real_release = utl_save._NewFile.release

        def refused(source, target):
            raise PermissionError(errno.EACCES, "Permission denied", source)

        def emptied(staged):
            real_release(staged)
            os.truncate(staged.path, 0)

        with self.renames_no_open_file(replace=refused), mock.patch.object(
            utl_save._NewFile, "release", emptied
        ):
            with self.assertRaises(OSError) as raised:
                self.save(b"new")
        self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
        self.assertEqual(raised.exception.errno, errno.ESTALE)
        self.assertEqual(self.read(), b"old")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_empties_its_closed_staged_file_once_removing_it_is_refused(self):
        # Opened again for reading alone, the staged file can't be emptied
        # through that; once removing it is refused, it is known to be there,
        # and it is emptied by name, so that what is left holds nothing.
        self.save(b"old")
        real_remove = os.remove

        def refused(source, target):
            raise PermissionError(errno.EACCES, "Permission denied", source)

        def staged_files_kept(path, *args):
            if os.path.basename(path).startswith(".dynaface-"):
                raise PermissionError(errno.EPERM, "Operation not permitted", path)
            return real_remove(path, *args)

        with self.renames_no_open_file(replace=refused, remove=staged_files_kept):
            with self.assertLogs("utl_save", "WARNING"):
                self.save(b"new contents")
        self.assertEqual(self.read(), b"new contents")
        left = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
        self.assertEqual(len(left), 1)
        self.assertEqual(os.path.getsize(os.path.join(self.folder, left[0])), 0)
        self.assertIn(self.folder, utl_save._STUCK_FOLDERS)

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_never_empties_by_name_a_file_put_in_the_staged_file_s_place(self):
        # By the time removing the staged file is refused, someone else who can
        # change the folder can have put another file in its place, which is
        # left as it is.
        self.save(b"old")
        real_remove = os.remove
        moved = os.path.join(self.other, "moved")

        def refused(source, target):
            raise PermissionError(errno.EACCES, "Permission denied", source)

        def swapped_and_refused(path, *args):
            if not os.path.basename(path).startswith(".dynaface-"):
                return real_remove(path, *args)
            os.rename(path, moved)
            with open(path, "wb") as f:
                f.write(b"another")
            raise PermissionError(errno.EPERM, "Operation not permitted", path)

        with self.renames_no_open_file(replace=refused, remove=swapped_and_refused):
            with self.assertLogs("utl_save", "WARNING"):
                self.save(b"new")
        self.assertEqual(self.read(), b"new")
        left = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
        self.assertEqual(len(left), 1)
        self.assertEqual(self.read(os.path.join(self.folder, left[0])), b"another")
        for path in (os.path.join(self.folder, left[0]), moved):
            os.remove(path)
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_never_empties_by_name_through_a_link_put_in_the_staged_file_s_place(
        self,
    ):
        # Someone else who can change the folder can move the closed staged file
        # and link it back in its place: what was moved is left as it is.
        path = os.path.join(self.folder, utl_save._staged_name())
        staged = utl_save._NewFile(path, open(path, "x+b"))
        staged.file.write(b"new")
        staged.file.flush()
        staged.release()
        moved = os.path.join(self.other, "moved")
        os.rename(path, moved)
        os.symlink(moved, path)
        with self.no_waiting():
            self.assertFalse(staged.empty(by_name=True))
        self.assertEqual(self.read(moved), b"new")
        for name in (path, moved):
            os.remove(name)

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_empties_its_closed_staged_file_with_another_link_by_name(self):
        # Closed to be renamed, the staged file can be linked elsewhere, and
        # stay closed should opening it again fail: the other link would keep
        # the new contents, so it is emptied by name before it is removed.
        self.save(b"old")
        extra = os.path.join(self.other, "extra")
        real_release = utl_save._NewFile.release
        real_reopen = utl_save._reopen
        reopened = []

        def linked(staged):
            real_release(staged)
            if not os.path.exists(extra):
                os.link(staged.path, extra)

        def fails_once(path, flags):
            reopened.append(flags & os.O_ACCMODE)
            if len(reopened) == 1:
                raise TimeoutError(errno.ETIMEDOUT, "Operation timed out", path)
            return real_reopen(path, flags)

        with self.renames_no_open_file(), mock.patch.object(
            utl_save._NewFile, "release", linked
        ), mock.patch.object(utl_save, "_reopen", fails_once):
            with self.assertRaises(OSError) as raised:
                self.save(b"new")
        self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
        self.assertEqual(raised.exception.errno, errno.ETIMEDOUT)
        self.assertEqual(reopened, [os.O_RDONLY, os.O_RDWR])
        self.assertEqual(self.read(), b"old")
        self.assertEqual(os.path.getsize(extra), 0)
        os.remove(extra)
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_empties_its_staged_file_when_a_refused_removal_leaves_it_unknown(self):
        # Closed once removing it is refused, the staged file is removed only
        # while its name names it. Should looking at the name fail, that can't
        # be told, and it is emptied by name instead, so that what is left
        # holds nothing.
        self.save(b"old")
        real_status = utl_save._status_through_file
        refused = []

        def refused_move(source, target):
            raise PermissionError(errno.EACCES, "Permission denied", source)

        def looked_at(path):
            if path in refused:
                refused.remove(path)
                raise OSError(errno.EIO, "Input/output error", path)
            return real_status(path)

        with self.renames_no_open_file(replace=refused_move):
            busy_remove = utl_save.os.remove

            def remove(path, *args):
                try:
                    return busy_remove(path, *args)
                except OSError as e:
                    if e.errno == errno.EBUSY:
                        refused.append(path)
                    raise

            with mock.patch.object(utl_save.os, "remove", remove), mock.patch.object(
                utl_save, "_status_through_file", looked_at
            ), self.assertLogs("utl_save", "WARNING"):
                self.save(b"new contents")
        self.assertEqual(self.read(), b"new contents")
        left = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
        self.assertEqual(len(left), 1)
        self.assertEqual(os.path.getsize(os.path.join(self.folder, left[0])), 0)
        self.assertIn(self.folder, utl_save._STUCK_FOLDERS)

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_never_opens_to_write_a_staged_file_gone_once_removing_it_is_refused(
        self,
    ):
        # Someone else can remove the staged file while its removal is refused
        # as busy. Once it is gone, it is never opened to write it, which would
        # lock it on a WebDAV volume (see
        # test_opens_its_closed_staged_file_to_write_only_where_it_is_known).
        self.save(b"old")
        real_reopen = utl_save._reopen
        modes = []

        def refused_move(source, target):
            raise PermissionError(errno.EACCES, "Permission denied", source)

        def reopened(path, flags):
            modes.append(flags & os.O_ACCMODE)
            return real_reopen(path, flags)

        with self.renames_no_open_file(replace=refused_move):
            busy_remove = utl_save.os.remove

            def removed_meanwhile(path, *args):
                try:
                    return busy_remove(path, *args)
                except OSError as e:
                    if e.errno == errno.EBUSY:
                        os.unlink(path)
                    raise

            with mock.patch.object(
                utl_save.os, "remove", removed_meanwhile
            ), mock.patch.object(utl_save, "_reopen", reopened), self.assertNoLogs(
                "utl_save", "WARNING"
            ):
                self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(modes, [os.O_RDONLY])
        self.assertEqual(utl_save._STUCK_FOLDERS, set())
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_opens_its_closed_staged_file_to_write_only_where_it_is_known(self):
        # A WebDAV volume can still show a staged file its server no longer has
        # (a rename reported failing that was done), and opening that to write
        # it locks it, which can make the server create it anew, or leave the
        # volume trying to read it until no file on it opens. A removal that
        # finds nothing tells no more of where it is, so it isn't opened then.
        self.save(b"old")
        real_reopen = utl_save._reopen
        modes = []

        def refused(source, target):
            raise PermissionError(errno.EACCES, "Permission denied", source)

        def not_found(path, *args):
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)

        def reopened(path, flags):
            modes.append(flags & os.O_ACCMODE)
            return real_reopen(path, flags)

        with self.renames_no_open_file(
            replace=refused, remove=not_found
        ), mock.patch.object(utl_save, "_reopen", reopened), self.assertNoLogs(
            "utl_save", "WARNING"
        ):
            self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(modes, [os.O_RDONLY])
        for name in os.listdir(self.folder):
            if name != "doc.dyfc":
                os.remove(os.path.join(self.folder, name))
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_opening_a_closed_staged_file_again_leaves_nothing_open_on_failing(
        self,
    ):
        # The file is opened without waiting, then told to wait when read; should
        # telling it fail, it is closed again.
        path = os.path.join(self.folder, "staged")
        with open(path, "wb"):
            pass
        real_open = os.open
        opened = []

        def recorded(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            return fd

        failing = OSError(errno.EIO, "Input/output error")
        with mock.patch.object(utl_save.os, "open", recorded), mock.patch.object(
            utl_save.os, "set_blocking", side_effect=failing
        ):
            with self.assertRaises(OSError):
                utl_save._reopen(path, os.O_RDONLY)
        self.assertEqual(len(opened), 1)
        with self.assertRaises(OSError) as raised:
            os.fstat(opened[0])
        self.assertEqual(raised.exception.errno, errno.EBADF)

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_writes_over_rather_than_rename_a_closed_staged_file_with_another_link(
        self,
    ):
        # A link to the staged file made elsewhere once it is closed to be
        # renamed would reach the document too.
        self.save(b"old")
        inode = self.inode()
        extra = os.path.join(self.other, "extra")
        real_release = utl_save._NewFile.release

        def linked(staged):
            real_release(staged)
            if not os.path.exists(extra):
                os.link(staged.path, extra)

        with self.renames_no_open_file(), mock.patch.object(
            utl_save._NewFile, "release", linked
        ):
            self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assertEqual(os.path.getsize(extra), 0)
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX open files")
    def test_never_removes_a_file_put_in_the_staged_file_s_place_once_it_is_closed(
        self,
    ):
        # Once the staged file is emptied and closed to be removed, someone else
        # who can change the folder can move it and put another file in its
        # place, which is left as it is.
        self.save(b"old")
        os.link(self.path, os.path.join(self.folder, "other.dyfc"))
        moved = os.path.join(self.other, "moved")
        real_release = utl_save._NewFile.release
        names = []

        def swapped(staged):
            real_release(staged)
            names.append(staged.path)
            os.rename(staged.path, moved)
            with open(staged.path, "wb") as f:
                f.write(b"another")

        with self.renames_no_open_file(), mock.patch.object(
            utl_save._NewFile, "release", swapped
        ):
            self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.read(names[0]), b"another")
        self.assertEqual(os.path.getsize(moved), 0)
        for path in (names[0], moved):
            os.remove(path)
        self.assert_nothing_left("other.dyfc")

    @unittest.skipIf(os.name == "nt", "POSIX")
    def test_only_removes_a_staged_file_closed_to_be_renamed(self):
        # Should renaming it fail, and opening it again too, the staged file is
        # removed by name while that names it still, never emptied by name.
        path = os.path.join(self.folder, utl_save._staged_name())
        staged = utl_save._NewFile(path, open(path, "x+b"))
        staged.release()
        utl_save._discard(staged, self.folder)
        self.assertEqual(os.listdir(self.folder), [])
        self.assertEqual(utl_save._STUCK_FOLDERS, set())

    @unittest.skipIf(os.name == "nt", "POSIX links and FIFOs")
    def test_never_moves_a_file_put_in_the_staged_file_s_place_once_it_is_closed(
        self,
    ):
        # Once the staged file is closed to be renamed, someone else who can
        # change the folder can move it and put another file in its place: a
        # file, a link to a private file of the user's, a hard link to one, or a
        # FIFO. The save neither moves that into the document's place, nor
        # reads, empties or waits on it, but fails, with the document as it was.
        # The files are of the staged file's size, so only which file each is
        # tells it from the staged file.
        real_release = utl_save._NewFile.release
        moved = os.path.join(self.other, "moved")

        def another_file(path):
            with open(path, "wb") as f:
                f.write(b"another")

        for case, put, code in (
            ("another file", another_file, errno.ESTALE),
            ("a link", lambda path: os.symlink(private, path), errno.ELOOP),
            ("a hard link", lambda path: os.link(private, path), errno.ESTALE),
            ("a FIFO", lambda path: os.mkfifo(path, 0o444), errno.ESTALE),
        ):
            with self.subTest(case):
                self.save(b"old")
                private = self.private_file()
                names = []

                def swapped(staged):
                    real_release(staged)
                    names.append(staged.path)
                    os.rename(staged.path, moved)
                    put(staged.path)

                with self.no_waiting(), self.renames_no_open_file(), mock.patch.object(
                    utl_save._NewFile, "release", swapped
                ):
                    with self.assertRaises(OSError) as raised:
                        self.save(b"new one")
                self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
                self.assertEqual(raised.exception.errno, code)
                self.assertEqual(self.read(), b"old")
                self.assert_untouched(private)
                if case == "another file":
                    self.assertEqual(self.read(names[0]), b"another")
                for path in (names[0], moved, private):
                    os.remove(path)
                self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_replaces_a_file_whose_acl_denies_changing_its_attributes(self):
        # The dates are copied before the ACL that would deny setting them.
        self.save(b"old")
        month_ago = int(time.time()) - 30 * 86400
        _set_date_added(self.path, month_ago)
        _add_acl_entry(self.path, "everyone deny writeattr")
        inode = self.inode()
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertNotEqual(self.inode(), inode)
        self.assertIn("0: group:everyone deny writeattr", _acl(self.path))
        self.assertEqual(_date_added(self.path), (month_ago, 0))
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_replaces_a_file_moved_into_a_folder_whose_new_files_cannot_be_removed(
        self,
    ):
        # The document doesn't carry the folder's entry (it was there before the
        # folder had it, as a moved-in file is), so it can be replaced, by a staged
        # file cleared of the entry it took from the folder.
        self.save(b"old")
        _add_acl_entry(self.folder, "everyone deny delete,file_inherit")
        for data in (b"first", b"second"):
            inode = self.inode()
            self.save(data)
            self.assertEqual(self.read(), data)
            self.assertNotEqual(self.inode(), inode)
        self.assertEqual(_acl(self.path), [])
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_keeps_the_dates_in_a_folder_whose_files_deny_changing_them(self):
        _add_acl_entry(self.folder, "everyone deny writeattr,file_inherit")
        self.save(b"old")
        added = _date_added(self.path)
        time.sleep(1.1)
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(_date_added(self.path), added)
        self.assertEqual(
            _acl(self.path), ["0: group:everyone inherited deny writeattr"]
        )
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_keeps_the_acl_exactly(self):
        # Entries the file took from a folder are kept, and none are taken from
        # the folder the file is in now.
        self.save(b"old")
        subprocess.run(["/bin/chmod", "+ai", "user:nobody deny read", self.path])
        _add_acl_entry(self.path, "user:daemon allow write")
        kept = _acl(self.path)
        _add_acl_entry(self.folder, "group:staff allow read,file_inherit")
        self.save(b"new")
        self.assertEqual(_acl(self.path), kept)
        self.assertIn("1: user:nobody inherited deny read", kept)

    @unittest.skipUnless(MACOS, "macOS ACLs")
    def test_replaces_a_file_on_a_volume_that_keeps_no_acls(self):
        # FAT and exFAT keep no ACLs, and setting even an empty one fails there,
        # which mustn't stop the replace.
        self.save(b"old")
        inode = self.inode()
        real = utl_save._libc()

        class NoAcls:
            def __getattr__(self, name):
                return getattr(real, name)

            def acl_get_file(self, *args):
                ctypes.set_errno(errno.ENOTSUP)
                return None

            def acl_set_file(self, *args):
                ctypes.set_errno(errno.ENOTSUP)
                return -1

            acl_get_fd_np = acl_get_file
            acl_set_fd_np = acl_set_file

        with mock.patch.object(utl_save, "_libc", return_value=NoAcls()):
            self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertNotEqual(self.inode(), inode)
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_a_private_file_takes_no_entries_from_its_folder(self):
        self.save(b"old")
        os.chmod(self.path, 0o600)
        _add_acl_entry(self.folder, "group:staff allow read,file_inherit")
        acls = []

        def write(file):
            acls.append(_acl(file.name))
            file.write(b"new")

        utl_save.save_file(self.path, write)
        self.assertEqual(acls, [[]])
        self.assertEqual(_acl(self.path), [])
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_a_staged_file_never_has_the_entries_its_folder_hands_new_files(self):
        # A file opened while it had them would stay open to whoever they let
        # open it, so it is created without them, not cleared of them after.
        # Where a network volume's server applies them, they come off at once.
        self.save(b"old")
        os.chmod(self.path, 0o600)
        _add_acl_entry(
            self.folder, "group:staff allow read,write,file_inherit,only_inherit"
        )
        handed = "0: group:staff inherited allow read,write"
        real_make_private = utl_save._make_private
        for by_server in (False, True):
            with self.subTest(by_server=by_server):
                created = []

                def make_private(staged):
                    created.append(_acl(staged.path))
                    return real_make_private(staged)

                with mock.patch.object(
                    utl_save, "_make_private", make_private
                ), mock.patch.object(
                    utl_save, "_hands_down_entries", return_value=not by_server
                ):
                    self.save(b"new")
                self.assertEqual(created, [[handed] if by_server else []])
                self.assertEqual(_acl(self.path), [])
                self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
                self.assert_nothing_left()

    @unittest.skipUnless(MACOS, "macOS volumes")
    def test_knows_which_volumes_hand_new_files_their_folder_s_entries(self):
        # The temp folder is on a local volume that keeps ACLs.
        self.assertTrue(utl_save._hands_down_entries(self.folder))
        for keeps_acls, by_server in ((0, 0), (1, 1)):
            values = {
                utl_save._PC_EXTENDED_SECURITY_NP: keeps_acls,
                utl_save._PC_AUTH_OPAQUE_NP: by_server,
            }
            with mock.patch.object(
                utl_save.os, "pathconf", side_effect=lambda f, n: values[n]
            ):
                self.assertFalse(utl_save._hands_down_entries(self.folder))

    @unittest.skipIf(os.name == "nt" or ROOT, "POSIX permissions")
    def test_saves_a_file_it_may_write_but_not_read(self):
        self.save(b"old")
        os.chmod(self.path, 0o200)
        try:
            self.save(b"new")
        finally:
            os.chmod(self.path, 0o600)
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS, "macOS file flags")
    def test_keeps_the_hidden_flag_where_the_name_decides_it(self):
        # FAT and exFAT volumes take a file's hidden flag from its name, so the
        # replace loses it, and it is set again.
        self.save(b"old")
        os.chflags(self.path, stat.UF_HIDDEN)
        real_replace = os.replace

        def replace_by_name(source, target):
            real_replace(source, target)
            os.chflags(target, 0)

        with mock.patch.object(utl_save.os, "replace", replace_by_name):
            self.save(b"new")
        self.assertEqual(os.stat(self.path).st_flags & stat.UF_HIDDEN, stat.UF_HIDDEN)
        self.assertEqual(self.read(), b"new")

    def test_stages_in_the_temp_folder_where_a_staged_file_was_left(self):
        self.save(b"old")
        real_remove = os.remove

        def refused_in_folder(path, *args):
            if os.path.dirname(path) == self.folder:
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return real_remove(path, *args)

        staged_in = []

        def write(file):
            staged_in.append(os.path.dirname(file.name))
            file.write(b"new contents")

        with mock.patch.object(
            utl_save.os, "replace", side_effect=refused_in_folder
        ), mock.patch.object(utl_save.os, "remove", side_effect=refused_in_folder):
            with self.assertLogs("utl_save", "WARNING"):
                for _ in range(3):
                    utl_save.save_file(self.path, write)
        self.assertEqual(staged_in, [self.folder, self.temp, self.temp])
        self.assertEqual(self.read(), b"new contents")
        left = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
        self.assertEqual(len(left), 1)
        self.assertEqual(os.listdir(self.temp), [])

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_writes_over_a_file_whose_extended_attributes_cannot_be_read(self):
        # A new file would lose what can't be copied, so the file is written
        # over instead.
        self.save(b"old")
        _add_acl_entry(self.path, "everyone deny readextattr")
        inode = self.inode()
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assertIn("0: group:everyone deny readextattr", _acl(self.path))
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_saves_a_file_whose_attributes_cannot_be_read(self):
        self.save(b"old")
        _add_acl_entry(self.path, "everyone deny readattr")
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_knows_a_file_whose_attributes_cannot_be_read_by_name(self):
        # An entry denying its attributes by name, as a document takes from a
        # folder that hands it out, lets the open file tell them: so the room the
        # old contents leave is known, and the file is written over only while
        # it is the one the save began with. Its ACL can't be read, so it is
        # written over rather than replaced, keeping the entry.
        for denied in ("readattr", "readsecurity"):
            with self.subTest(denied):
                self.save(b"old")
                _add_acl_entry(self.path, f"everyone deny {denied}")
                full = mock.Mock(total=1 << 30, free=0)
                with mock.patch.object(
                    utl_save.shutil, "disk_usage", return_value=full
                ):
                    with self.assertRaises(OSError) as raised:
                        self.save(b"longer contents")
                self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
                self.assertEqual(raised.exception.errno, errno.ENOSPC)
                self.assertEqual(self.read(), b"old")

                private = self.private_file()
                moved = os.path.join(self.other, "moved.dyfc")

                def write(file):
                    file.write(b"new")
                    os.rename(self.path, moved)
                    os.link(private, self.path)

                with self.assertRaises(OSError):
                    utl_save.save_file(self.path, write)
                self.assert_untouched(private)
                os.remove(self.path)
                os.rename(moved, self.path)

                inode = self.inode_through_file()
                self.save(b"new")
                self.assertEqual(self.read(), b"new")
                self.assertEqual(self.inode_through_file(), inode)
                subprocess.run(["/bin/chmod", "-N", self.path], check=True)
                self.assertEqual(self.inode(), inode)
                os.remove(private)
                self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_reuses_no_room_where_nothing_is_known_of_a_file(self):
        # A file the user may write but not read, whose attributes can't be read
        # either, is written over knowing nothing of it, so contents that don't
        # fit in the free room alone are refused.
        self.save(b"old")
        contents = b"new contents"
        os.chmod(self.path, 0o200)
        _add_acl_entry(self.path, "everyone deny readattr")
        try:
            room = mock.Mock(total=1 << 30, free=len(contents) - 1)
            with mock.patch.object(utl_save.shutil, "disk_usage", return_value=room):
                with self.assertRaises(OSError) as raised:
                    self.save(contents)
            self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
            self.assertEqual(raised.exception.errno, errno.ENOSPC)
            self.save(contents)
        finally:
            subprocess.run(["/bin/chmod", "-N", self.path], check=True)
            os.chmod(self.path, 0o600)
        self.assertEqual(self.read(), contents)
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_writes_over_a_file_nothing_is_known_of_through_the_one_it_began_with(
        self,
    ):
        # Nothing tells such a file from another put in its place, so it is
        # opened when the save begins: someone else who can change the folder,
        # moving it away during the save and linking a private file of the
        # user's in its place, can't have the save write into that file.
        self.save(b"old")
        os.chmod(self.path, 0o200)
        _add_acl_entry(self.path, "everyone deny readattr")
        private = self.private_file()
        moved = os.path.join(self.other, "moved.dyfc")

        def write(file):
            file.write(b"new")
            os.rename(self.path, moved)
            os.link(private, self.path)

        try:
            utl_save.save_file(self.path, write)
        finally:
            subprocess.run(["/bin/chmod", "-N", moved], check=True)
            os.chmod(moved, 0o600)
        self.assert_untouched(private)
        self.assertEqual(self.read(moved), b"new")
        self.assert_nothing_left()

    @unittest.skipIf(os.name == "nt", "POSIX links")
    def test_never_opens_a_link_put_in_place_of_a_file_nothing_is_known_of(self):
        # Nor a link that someone else put in its place just as the save began:
        # the save is refused.
        self.save(b"old")
        private = self.private_file()
        moved = os.path.join(self.other, "moved.dyfc")
        real_check = utl_save._check_writable

        def swapped(target, filename):
            real_check(target, filename)
            os.rename(self.path, moved)
            os.symlink(private, self.path)

        with mock.patch.object(
            utl_save, "_stat", return_value=(True, None)
        ), mock.patch.object(utl_save, "_check_writable", swapped):
            with self.assertRaises(OSError) as raised:
                self.save(b"new")
        self.assertEqual(raised.exception.errno, errno.ELOOP)
        self.assert_untouched(private)
        self.assertEqual(self.read(moved), b"old")
        os.remove(self.path)
        os.rename(moved, self.path)
        os.remove(private)
        self.assert_nothing_left()

    @unittest.skipUnless(MACOS and not ROOT, "macOS ACLs")
    def test_never_waits_on_a_fifo_in_the_document_s_place(self):
        # Opening a FIFO to read it waits for something to write to it, forever,
        # unless told not to wait. Someone else who can change the folder can
        # put one in the document's place during the save, which is refused; or
        # before it, denying its attributes by name, so that they are read
        # through it opened.
        self.save(b"old")
        moved = os.path.join(self.other, "moved.dyfc")

        def write(file):
            file.write(b"new")
            os.rename(self.path, moved)
            os.mkfifo(self.path, 0o644)

        with self.no_waiting(), self.assertRaises(OSError):
            utl_save.save_file(self.path, write)
        self.assertTrue(stat.S_ISFIFO(os.lstat(self.path).st_mode))
        self.assertEqual(self.read(moved), b"old")
        self.assertEqual(os.listdir(self.temp), [])

        os.remove(self.path)
        os.mkfifo(self.path, 0o644)
        _add_acl_entry(self.path, "everyone deny readattr")
        with self.no_waiting(), contextlib.suppress(OSError):
            self.save(b"new")
        subprocess.run(["/bin/chmod", "-N", self.path], check=True)
        self.assertEqual(os.listdir(self.temp), [])

    def inode_through_file(self):
        """The file's number, read through the open file, as its name may not
        tell it."""
        with open(self.path, "rb") as f:
            return os.fstat(f.fileno()).st_ino

    def test_a_staged_file_that_cannot_be_removed_is_emptied(self):
        self.save(b"old")

        def refused(*args):
            raise PermissionError(errno.EPERM, "Operation not permitted")

        with mock.patch.object(
            utl_save.os, "replace", side_effect=refused
        ), mock.patch.object(utl_save.os, "remove", side_effect=refused):
            with self.assertLogs("utl_save", "WARNING"):
                self.save(b"new contents")
        self.assertEqual(self.read(), b"new contents")
        left = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
        self.assertEqual(len(left), 1)
        self.assertEqual(os.path.getsize(os.path.join(self.folder, left[0])), 0)

    def test_a_failed_write_leaves_an_empty_staged_file_where_it_cannot_be_removed(
        self,
    ):
        # What the file's buffer held when write failed is dropped, not written
        # out when the file is closed.
        self.save(b"old")
        real_remove = os.remove

        def refused(path, *args):
            if os.path.basename(path).startswith(".dynaface-"):
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return real_remove(path, *args)

        with mock.patch.object(utl_save.os, "remove", side_effect=refused):
            with self.assertLogs("utl_save", "WARNING"):
                with self.assertRaises(ValueError):
                    utl_save.save_file(self.path, _fail)
        self.assertEqual(self.read(), b"old")
        left = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
        self.assertEqual(len(left), 1)
        self.assertEqual(os.path.getsize(os.path.join(self.folder, left[0])), 0)

    def test_windows_refuses_a_file_it_cannot_open_for_writing(self):
        # os.access() there reads only the read-only attribute, not an ACL.
        self.save(b"old")
        real_open = os.open

        def denied(path, flags, *args):
            if path == self.path and flags & os.O_WRONLY:
                raise PermissionError(errno.EACCES, "Access is denied", path)
            return real_open(path, flags, *args)

        with mock.patch.object(utl_save, "_WINDOWS", True), mock.patch.object(
            utl_save.os, "open", side_effect=denied
        ):
            with self.assertRaises(PermissionError):
                self.save(b"new")
        self.assertEqual(self.read(), b"old")
        self.assert_nothing_left()

    def test_windows_writes_over_hidden_system_compressed_or_encrypted_files(self):
        attributes = {
            "hidden": stat.FILE_ATTRIBUTE_HIDDEN,
            "system": stat.FILE_ATTRIBUTE_SYSTEM,
            "compressed": stat.FILE_ATTRIBUTE_COMPRESSED,
            "encrypted": stat.FILE_ATTRIBUTE_ENCRYPTED,
        }
        real_stat = utl_save._stat
        for name, attribute in attributes.items():
            with self.subTest(name):
                self.save(b"old")
                inode = self.inode()

                def with_attribute(target, attribute=attribute):
                    exists, status = real_stat(target)
                    return exists, _WithAttributes(status, attribute)

                with mock.patch.object(utl_save, "_stat", with_attribute):
                    self.save(b"new")
                self.assertEqual(self.read(), b"new")
                self.assertEqual(self.inode(), inode)
                self.assert_nothing_left()

    def test_windows_replaces_a_file_its_folder_encrypts_new_files_like(self):
        # In a folder that encrypts (or compresses) new files the staged file is
        # encrypted too, so it can take the document's place.
        real_stat = utl_save._stat
        for name in ("compressed", "encrypted"):
            attribute = getattr(stat, f"FILE_ATTRIBUTE_{name.upper()}")
            with self.subTest(name):
                self.save(b"old")
                inode = self.inode()

                def with_attribute(target, attribute=attribute):
                    exists, status = real_stat(target)
                    return exists, _WithAttributes(status, attribute)

                with mock.patch.object(
                    utl_save, "_stat", with_attribute
                ), mock.patch.object(
                    utl_save, "_file_attributes", return_value=attribute
                ):
                    self.save(b"new")
                self.assertEqual(self.read(), b"new")
                self.assertNotEqual(self.inode(), inode)
                self.assert_nothing_left()

    def test_windows_stages_an_encrypted_document_in_memory_rather_than_in_clear(
        self,
    ):
        # A Windows file is encrypted only when its folder encrypts new files, so
        # the staged copy of a document encrypted on its own, or of one whose
        # folder's staged file was left, would hold it in the clear.
        real_stat = utl_save._stat

        def encrypted(target):
            exists, status = real_stat(target)
            return exists, _WithAttributes(status, stat.FILE_ATTRIBUTE_ENCRYPTED)

        for stuck in (False, True):
            with self.subTest(stuck=stuck):
                self.save(b"old")
                if stuck:
                    utl_save._STUCK_FOLDERS.add(self.folder)
                inode = self.inode()
                files = []

                def write(file):
                    files.append(file)
                    file.write(b"new")

                with mock.patch.object(utl_save, "_stat", encrypted):
                    utl_save.save_file(self.path, write)
                self.assertIsInstance(files[0], io.BytesIO)
                self.assertEqual(self.read(), b"new")
                self.assertEqual(self.inode(), inode)
                self.assert_nothing_left()


class _Module:
    """A module with some attributes replaced, and some taken away."""

    def __init__(self, module, missing=(), **replaced):
        self._module = module
        self._missing = set(missing)
        self.__dict__.update(replaced)

    def __getattr__(self, name):
        if name in self._missing:
            raise AttributeError(name)
        return getattr(self._module, name)


class WindowsSaveTest(_SaveTest):
    """
    Saves as on Windows, with its rule for open files: no one can rename or
    remove a file someone holds open, as CPython opens files without
    FILE_SHARE_DELETE, so os.replace() and os.remove() of a path the save, or
    someone else, holds open fail with PermissionError, as MoveFileExW() and
    DeleteFileW() do; and one held by someone who lets others only read it
    can't be opened to write it either. Windows has no file owners or
    pathconf() here, and os.chmod() takes no open file before Python 3.13. A
    volume can number no files, as WebDAV volumes and Google Drive don't, giving
    each the number 0 (self.numbered).
    """

    def setUp(self):
        super().setUp()
        self.opened = []  # (the real path, the file) the save opened by name
        self.held = set()  # paths someone else holds open
        self.reading = set()  # those of them others may only read
        self.numbered = True
        self.sleeps = []

    def is_open(self, path):
        path = os.path.realpath(path)
        return path in self.held or any(
            name == path and not file.closed for name, file in self.opened
        )

    def windows(self, replace=os.replace, remove=os.remove, sleep=None, **calls):
        def tracked_open(file, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if isinstance(file, str) and set(mode) & set("wax+"):
                if os.path.realpath(file) in self.reading:
                    raise PermissionError(errno.EACCES, "The file is in use", file)
            opened = builtins.open(file, *args, **kwargs)
            if isinstance(file, str):
                self.opened.append((os.path.realpath(file), opened))
            return opened

        def status(real):
            def numbered_or_not(*args, **kwargs):
                found = real(*args, **kwargs)
                return found if self.numbered else _Unnumbered(found)

            return numbered_or_not

        def windows_replace(source, target):
            for path in (source, target):
                if self.is_open(path):
                    raise PermissionError(errno.EACCES, "The file is in use", path)
            replace(source, target)

        def windows_remove(path, *args):
            if self.is_open(path):
                raise PermissionError(errno.EACCES, "The file is in use", path)
            remove(path, *args)

        def by_name(*args):
            raise AssertionError("a file was truncated by name")

        windows_os = _Module(
            os,
            missing=("geteuid", "pathconf"),
            supports_fd=os.supports_fd - {os.chmod},
            replace=windows_replace,
            remove=windows_remove,
            truncate=by_name,
            stat=status(os.stat),
            lstat=status(os.lstat),
            fstat=status(os.fstat),
            **calls,
        )
        for patcher in (
            mock.patch.object(utl_save, "_WINDOWS", True),
            mock.patch.object(utl_save, "_NOFOLLOW", 0),
            mock.patch.object(utl_save, "_NONBLOCK", 0),
            mock.patch.object(utl_save, "os", windows_os),
            mock.patch.object(utl_save, "sys", _Module(sys, platform="win32")),
            mock.patch.object(utl_save, "open", tracked_open, create=True),
            mock.patch.object(
                utl_save, "time", _Module(time, sleep=sleep or self.sleeps.append)
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_replaces_a_file_with_a_new_one(self):
        self.windows()
        self.save(b"old")
        inode = self.inode()
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertNotEqual(self.inode(), inode)
        self.assertEqual(self.sleeps, [])
        self.assert_nothing_left()

    def test_writes_over_a_file_it_cannot_replace(self):
        # A backup tool or a scanner holding the document refuses the replace.
        # On a volume that numbers no files, too, that must not pass for a
        # replace done.
        self.windows()
        for numbered in (True, False):
            with self.subTest(numbered=numbered):
                self.numbered = numbered
                self.save(b"old")
                inode = self.inode()
                self.sleeps.clear()
                self.held.add(self.path)
                self.save(b"new")
                self.held.clear()
                self.assertEqual(self.read(), b"new")
                self.assertEqual(self.inode(), inode)
                self.assertEqual(len(self.sleeps), utl_save._ATTEMPTS - 1)
                self.assert_nothing_left()

    def test_writes_over_a_file_whose_staged_copy_takes_no_permissions(self):
        # The staged file was never renamed then, whichever file the volume
        # numbers it, or none.
        def chmod(path, *args, **kwargs):
            raise PermissionError(errno.EACCES, "Access is denied", path)

        self.windows(chmod=chmod)
        for numbered in (True, False):
            with self.subTest(numbered=numbered):
                self.numbered = numbered
                self.save(b"old")
                inode = self.inode()
                self.save(b"new")
                self.assertEqual(self.read(), b"new")
                self.assertEqual(self.inode(), inode)
                self.assert_nothing_left()

    def test_writes_over_where_a_scanner_lets_the_staged_file_only_be_read(self):
        # A scanner that opens the staged file once it is closed, letting others
        # only read it, refuses the replace and opening the file again to write
        # it: it is opened again for reading alone, so it can't be emptied, and
        # it is left while the scanner holds it.
        def scanned(source, target):
            for held in (self.held, self.reading):
                held.add(os.path.realpath(source))
            raise PermissionError(errno.EACCES, "The file is in use", source)

        self.windows(replace=scanned)
        self.save(b"old")
        inode = self.inode()
        with self.assertLogs("utl_save", "WARNING"):
            self.save(b"new contents")
        self.assertEqual(self.read(), b"new contents")
        self.assertEqual(self.inode(), inode)
        left = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
        self.assertEqual(len(left), 1)
        self.assertEqual(self.read(os.path.join(self.folder, left[0])), b"new contents")
        self.assertIn(self.folder, utl_save._STUCK_FOLDERS)

    def test_stages_beside_a_document_in_a_deep_folder(self):
        # Windows, unless long paths are turned on, refuses a path of 260
        # characters or more; the staged file's name leaves room for a folder's
        # path of 232 of them.
        deep = os.path.join(self.folder, "d" * (231 - len(self.folder)))
        os.mkdir(deep)
        path = os.path.join(deep, "visit.dyfc")

        def short_paths_only(name, *args, **kwargs):
            if len(os.fsdecode(name)) >= 260:
                raise FileNotFoundError(
                    errno.ENOENT, "The system cannot find the path specified", name
                )
            return os.open(name, *args, **kwargs)

        self.windows(open=short_paths_only)
        self.save(b"old", path)
        inode = os.stat(path).st_ino
        self.save(b"new", path)
        self.assertEqual(self.read(path), b"new")
        self.assertNotEqual(os.stat(path).st_ino, inode)
        self.assertEqual(os.listdir(deep), ["visit.dyfc"])
        self.assertEqual(os.listdir(self.temp), [])

    def test_a_failed_write_leaves_the_file_as_it_was(self):
        self.windows()
        self.save(b"old")
        with self.assertRaises(ValueError):
            utl_save.save_file(self.path, _fail)
        self.assertEqual(self.read(), b"old")
        self.assert_nothing_left()
        self.assertNotIn(self.folder, utl_save._STUCK_FOLDERS)

    def test_a_failed_new_file_leaves_nothing(self):
        self.windows()
        with self.assertRaises(ValueError):
            utl_save.save_file(self.path, _fail)
        self.assertEqual(os.listdir(self.folder), [])
        self.assertEqual(os.listdir(self.temp), [])

    def test_a_staged_file_that_cannot_be_removed_is_emptied_through_the_open_file(
        self,
    ):
        self.save(b"old")

        def refused(*args):
            raise PermissionError(errno.EACCES, "Access is denied")

        self.windows(replace=refused, remove=refused)
        for numbered in (True, False):
            with self.subTest(numbered=numbered):
                self.numbered = numbered
                contents = b"new contents %d" % numbered
                with self.assertLogs("utl_save", "WARNING"):
                    self.save(contents)
                self.assertEqual(self.read(), contents)
                left = [name for name in os.listdir(self.folder) if name != "doc.dyfc"]
                self.assertEqual(len(left), 1)
                self.assertEqual(os.path.getsize(os.path.join(self.folder, left[0])), 0)
                self.assertIn(self.folder, utl_save._STUCK_FOLDERS)
                os.remove(os.path.join(self.folder, left[0]))
                utl_save._STUCK_FOLDERS.discard(self.folder)

    def test_never_touches_a_file_linked_in_place_of_the_closed_staged_file(self):
        # Windows closes the staged file to rename it. While the replace is
        # refused (a backup tool holding the document), someone else who can
        # change the folder can rename the closed file and link a private file
        # of the user's in its place. The save must neither read that file into
        # the document nor empty it, whether or not the link can be removed: by
        # which file it is, though it is of the staged file's size, and where
        # the volume numbers no files, by its size, which is another then.
        private = self.private_file()
        moved = os.path.join(self.other, "moved")
        staged = []
        removable = []
        contents = []

        def remove(path, *args):
            if staged and path == staged[0] and not removable[-1]:
                raise PermissionError(errno.EACCES, "Access is denied", path)
            os.remove(path, *args)

        def swap(delay):
            self.sleeps.append(delay)
            if len(self.sleeps) == 1:
                os.rename(staged[0], moved)
                os.link(private, staged[0])

        def write(file):
            staged.append(file.name)
            file.write(contents[-1])

        self.windows(remove=remove, sleep=swap)
        for numbered, can_remove in (
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ):
            with self.subTest(numbered=numbered, can_remove=can_remove):
                self.numbered = numbered
                removable.append(can_remove)
                contents.append(b"new one" if numbered else b"new")
                self.save(b"old")
                staged.clear()
                self.sleeps.clear()
                self.held.add(self.path)
                logged = (
                    contextlib.nullcontext()
                    if can_remove
                    else self.assertLogs("utl_save", "WARNING")
                )
                with logged, self.assertRaises(OSError) as raised:
                    utl_save.save_file(self.path, write)
                self.held.clear()
                self.assertNotIsInstance(raised.exception, utl_save.IncompleteSaveError)
                self.assertEqual(self.read(), b"old")
                self.assert_untouched(private)
                self.assertEqual(self.read(moved), contents[-1])
                os.remove(moved)
                if not can_remove:
                    os.remove(staged[0])
                    utl_save._STUCK_FOLDERS.discard(self.folder)
                self.assert_nothing_left()

    def test_keeps_a_replace_reported_failing_that_was_done(self):
        # A network volume can report a rename failing that it did; the closed
        # staged file is known in the document's place by which file it is.
        self.save(b"old")
        inode = self.inode()

        def done_but_failed(source, target):
            os.replace(source, target)
            raise OSError(errno.EIO, "Input/output error", source)

        self.windows(replace=done_but_failed)
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertNotEqual(self.inode(), inode)
        self.assert_nothing_left()

    def test_writes_over_a_file_whose_replace_is_reported_busy(self):
        # The staged file is closed to be renamed already, so a replace
        # reported busy is refused as any other is.
        def busy(source, target):
            raise OSError(errno.EBUSY, "Resource busy", source)

        self.windows(replace=busy)
        self.save(b"old")
        inode = self.inode()
        self.save(b"new")
        self.assertEqual(self.read(), b"new")
        self.assertEqual(self.inode(), inode)
        self.assertEqual(self.sleeps, [])
        self.assert_nothing_left()

    def test_keeps_the_new_contents_when_a_replace_removed_the_file_and_failed(self):
        # As a WebDAV server can, which carries out the one request Windows
        # sends in two steps; whichever file the volume numbers the staged file,
        # or none.
        real_remove = os.remove

        def removed_then_failed(source, target):
            with contextlib.suppress(FileNotFoundError):
                real_remove(target)
            raise OSError(errno.EIO, "Input/output error", source)

        self.windows(replace=removed_then_failed)
        for numbered in (True, False):
            with self.subTest(numbered=numbered):
                self.numbered = numbered
                self.save(b"old")
                with self.assertRaises(utl_save.MissingFileError) as raised:
                    self.save(b"new")
                self.assertEqual(self.read(raised.exception.kept), b"new")
                self.assertFalse(os.path.lexists(self.path))
                os.remove(raised.exception.kept)
                self.assertEqual(os.listdir(self.folder), [])

    def test_a_replace_reported_failing_shows_as_failing_where_files_have_no_number(
        self,
    ):
        # The staged file can't be known in the document's place then, so the
        # save reports failing, whatever the sizes, though the document has the
        # new contents: it says the document may hold them, and keeps nothing,
        # as the staged file is the document now.
        def done_but_failed(source, target):
            os.replace(source, target)
            raise OSError(errno.EIO, "Input/output error", source)

        self.windows(replace=done_but_failed)
        for contents in (b"new", b"newer contents"):
            with self.subTest(contents=contents):
                self.numbered = True
                self.save(b"old")
                self.numbered = False
                with self.assertRaises(utl_save.MissingFileError) as raised:
                    self.save(contents)
                self.assertEqual(raised.exception.errno, errno.ENOENT)
                self.assertIsNone(raised.exception.kept)
                self.assertEqual(self.read(), contents)
                self.assert_nothing_left()

    def test_keeps_the_new_contents_when_the_file_lost_its_own_after_a_replace(self):
        # A WebDAV server that removed the file and failed can show it as an
        # empty one: where the volume numbers no files, its size tells it, even
        # after replaces refused busy, as without a number it can't be told
        # from another computer's version of the file.
        full = mock.Mock(total=1 << 30, free=0)

        def emptied_then_failed(source, target):
            os.truncate(target, 0)
            raise OSError(code, os.strerror(code), source)

        self.windows(replace=emptied_then_failed)
        for numbered, code in (
            (True, errno.EIO),
            (False, errno.EIO),
            (False, errno.EBUSY),
        ):
            with self.subTest(numbered=numbered, errno=errno.errorcode[code]):
                self.numbered = numbered
                with open(self.path, "wb") as f:
                    f.write(b"old")
                with mock.patch.object(
                    utl_save.shutil, "disk_usage", return_value=full
                ), self.assertRaises(utl_save.MissingFileError) as raised:
                    self.save(b"longer contents")
                self.assertEqual(raised.exception.errno, errno.ENOSPC)
                self.assertEqual(self.read(raised.exception.kept), b"longer contents")
                os.remove(raised.exception.kept)
                self.assertEqual(self.read(), b"")
                self.assert_nothing_left()


class _Unnumbered:
    """A file's status on a volume that numbers no files, as Windows reports one
    on some (WebDAV volumes, Google Drive)."""

    st_dev = 0
    st_ino = 0

    def __init__(self, status):
        self._status = status

    def __getattr__(self, name):
        return getattr(self._status, name)


class _FailsToClose:
    """An open file whose closing fails once it is closed, as a network volume's
    can for a write that failed."""

    def __init__(self, file):
        self._file = file
        self.raw = _RawFailsToClose(file.raw)

    def __getattr__(self, name):
        return getattr(self._file, name)


class _RawFailsToClose:
    """The unbuffered file of a _FailsToClose."""

    def __init__(self, raw):
        self._raw = raw

    def close(self):
        self._raw.close()
        raise OSError(errno.EIO, "Input/output error")

    def __getattr__(self, name):
        return getattr(self._raw, name)


class _WithAttributes:
    """A file's status with Windows file attributes added."""

    def __init__(self, status, attributes):
        self._status = status
        self.st_file_attributes = attributes

    def __getattr__(self, name):
        return getattr(self._status, name)


if __name__ == "__main__":
    unittest.main()
