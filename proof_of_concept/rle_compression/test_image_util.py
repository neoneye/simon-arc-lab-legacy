import unittest
import numpy as np
from image_util import image_create

class TestImageUtil(unittest.TestCase):
    def test_image_fill(self):
        actual = image_create(2, 3, 4)

        expected = np.zeros((3, 2), dtype=np.uint8)
        expected[0:3, 0:2] = [
            [4, 4],
            [4, 4],
            [4, 4]]

        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
