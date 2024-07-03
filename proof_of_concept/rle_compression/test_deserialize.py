import unittest
import numpy as np
from deserialize import deserialize

class TestDeserialize(unittest.TestCase):
    def test_deserialize_full(self):
        a = "11 11 0,0c2e0,02a12e0,,,,,,0c2e0,0,"
        actual = deserialize(a)
        # print(actual)

        # Create a 11x11 array filled with zeros
        expected = np.zeros((11, 11), dtype=np.uint8)

        # Fill in the specific values
        expected[1:9, 1:5] = [
            [2, 2, 2, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 2, 2, 2]]
        # print(expected)
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
