import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynaface.util import is_zero_tuple


import unittest


class TestIsZeroTuple(unittest.TestCase):

    def test_exact_zeros(self):
        self.assertTrue(is_zero_tuple((0, 0)))

    def test_float_zeros(self):
        self.assertTrue(is_zero_tuple((0.0, -0.0, 0.0, 0.0)))

    def test_near_zeros(self):
        self.assertTrue(is_zero_tuple((1e-10, 0.0, 0.0, -1e-10)))

    def test_above_tolerance(self):
        self.assertFalse(is_zero_tuple((1e-6, 0.0, 0.0, 0.0), tol=1e-9))

    def test_wrong_structure(self):
        self.assertFalse(is_zero_tuple("not a tuple"))
        self.assertFalse(is_zero_tuple(["list", "of", "strings"]))
        self.assertFalse(is_zero_tuple(123))


if __name__ == "__main__":
    unittest.main()
