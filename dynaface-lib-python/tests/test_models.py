"""Tests for downloading the models and choosing a device, without the network."""

import hashlib
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import requests

from dynaface import models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_FILES = ["blaze_face_short_range.onnx", "spiga_wflw.onnx", "u2net.onnx"]
ZIP_URL = "https://example.com/models.zip"


def make_zip():
    """A stand-in for the model bundle: the three files, each a few bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name in MODEL_FILES:
            bundle.writestr(name, f"not really {name}")
    return buffer.getvalue()


ZIP = make_zip()
ZIP_SHA256 = hashlib.sha256(ZIP).hexdigest()


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status = status
        self.payload = payload
        self.content = content

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


class FakeWeb:
    """Answers requests.get() from a table of URLs, and records what was asked."""

    def __init__(self, routes):
        self.routes = routes
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        answer = self.routes.get(url, FakeResponse(status=404))
        if isinstance(answer, Exception):
            raise answer
        return answer


REDIRECT_OK = FakeResponse(payload={models.MODEL_VERSION: {"url": ZIP_URL}})


class TestDownloadModels(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name) / "models"

    def download(self, routes, verify_hash=True, expected_hash=ZIP_SHA256):
        web = FakeWeb(routes)
        with mock.patch.object(models.requests, "get", web.get), mock.patch.object(
            models, "EXPECTED_SHA256", expected_hash
        ):
            result = models.download_models(self.folder, verify_hash=verify_hash)
        return result, web.requested

    def assert_unpacked(self):
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), MODEL_FILES)

    def test_models_already_there(self):
        self.folder.mkdir(parents=True)
        (self.folder / "spiga_wflw.onnx").write_bytes(b"")
        web = FakeWeb({})
        with mock.patch.object(models.requests, "get", web.get):
            # A folder given as a string works as well as a Path.
            self.assertEqual(models.download_models(str(self.folder)), str(self.folder))
        self.assertEqual(web.requested, [])

    def test_download_from_the_redirect(self):
        result, requested = self.download(
            {models.REDIRECT_URL: REDIRECT_OK, ZIP_URL: FakeResponse(content=ZIP)}
        )
        self.assertEqual(result, str(self.folder))
        self.assertEqual(requested, [models.REDIRECT_URL, ZIP_URL])
        self.assert_unpacked()  # and the zip itself is gone

    def test_redirect_fails(self):
        _, requested = self.download(
            {
                models.REDIRECT_URL: requests.ConnectionError("offline"),
                models.FALLBACK_URL: FakeResponse(content=ZIP),
            }
        )
        self.assertEqual(requested, [models.REDIRECT_URL, models.FALLBACK_URL])
        self.assert_unpacked()

    def test_download_fails_then_fallback_works(self):
        _, requested = self.download(
            {
                models.REDIRECT_URL: REDIRECT_OK,
                ZIP_URL: FakeResponse(status=503),
                models.FALLBACK_URL: FakeResponse(content=ZIP),
            }
        )
        self.assertEqual(requested[-1], models.FALLBACK_URL)
        self.assert_unpacked()

    def test_both_downloads_fail(self):
        # The first failure is the one reported.
        routes = {
            models.REDIRECT_URL: REDIRECT_OK,
            ZIP_URL: FakeResponse(status=503),
            models.FALLBACK_URL: FakeResponse(status=404),
        }
        with self.assertRaisesRegex(requests.HTTPError, "503"):
            self.download(routes)

    def test_fallback_alone_fails(self):
        routes = {
            models.REDIRECT_URL: FakeResponse(status=500),
            models.FALLBACK_URL: FakeResponse(status=404),
        }
        with self.assertRaisesRegex(requests.HTTPError, "404"):
            self.download(routes)

    def test_hash_mismatch(self):
        routes = {models.REDIRECT_URL: REDIRECT_OK, ZIP_URL: FakeResponse(content=ZIP)}
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.download(routes, expected_hash="0" * 64)
        self.assertEqual(list(self.folder.iterdir()), [])  # the bad zip is deleted

    def test_hash_check_skipped(self):
        routes = {models.REDIRECT_URL: REDIRECT_OK, ZIP_URL: FakeResponse(content=ZIP)}
        self.download(routes, verify_hash=False, expected_hash="0" * 64)
        self.assert_unpacked()


class TestDevices(unittest.TestCase):
    def detect(self, providers, system):
        with mock.patch.object(
            models.ort, "get_available_providers", return_value=providers
        ), mock.patch.object(models.platform, "system", return_value=system):
            return models.detect_device()

    def test_detect_device(self):
        cuda = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        coreml = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(self.detect(cuda, "Linux"), "cuda")
        self.assertEqual(self.detect(coreml, "Darwin"), "mps")
        self.assertEqual(self.detect(coreml, "Linux"), "cpu")
        self.assertEqual(self.detect(["CPUExecutionProvider"], "Windows"), "cpu")

    def test_loading_needs_a_folder(self):
        with mock.patch.object(models, "_model_path", None):
            with self.assertRaises(ValueError):
                models._init_onnx()
