import unittest
from facial_analysis.image import *

class TestImage(unittest.TestCase):
    def test_load_image(self):
        img = load_image("./tests_data/img1-512.jpg")
        assert(img.shape[0]==512)
        assert(img.shape[1]==512)

if __name__ == '__main__':
    unittest.main()


