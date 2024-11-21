import unittest
import numpy as np
from .image_to_string import *

class TestImageToString(unittest.TestCase):
    def test_simple(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6]], dtype=np.uint8)
        actual = image_to_string(image)
        expected = "123\n456"
        self.assertEqual(actual, expected)
