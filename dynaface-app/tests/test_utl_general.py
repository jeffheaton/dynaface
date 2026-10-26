"""Tests for assert_standard_python, which checks what a document can hold.

Run from dynaface-app with: python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np

from utl_general import assert_standard_python


class _NoToList(np.ndarray):
    def tolist(self):
        raise AssertionError("a numeric array was walked element by element")


class AssertStandardPythonTest(unittest.TestCase):
    def test_accepts_numeric_arrays_without_walking_them(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8).view(_NoToList)
        assert_standard_python({"frames": [[image, np.arange(3.0).view(_NoToList)]]})

    def test_checks_the_elements_of_object_arrays(self):
        values = np.empty(2, dtype=object)
        values[0], values[1] = 1.5, object()
        with self.assertRaises(TypeError):
            assert_standard_python({"frames": [values]})

    def test_accepts_object_arrays_of_standard_values(self):
        values = np.empty(2, dtype=object)
        values[0], values[1] = 1.5, (1, "x")
        assert_standard_python(values)

    def test_rejects_other_objects(self):
        with self.assertRaises(TypeError):
            assert_standard_python({"frames": [[object()]]})


if __name__ == "__main__":
    unittest.main()
