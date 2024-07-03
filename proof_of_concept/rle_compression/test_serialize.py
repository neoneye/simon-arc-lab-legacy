import unittest
import numpy as np
from serialize import serialize

class TestSerialize(unittest.TestCase):
    def test_serialize_full(self):
        # Create a 11x11 array filled with zeros
        input = np.zeros((11, 11), dtype=np.uint8)

        # Fill in the specific values
        input[1:9, 1:5] = [
            [2, 2, 2, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 2, 2, 2]]
        # print(input)

        actual = serialize(input)
        # print(actual)
        expected = "11 11 0,0c2e0,02a12e0,,,,,,0c2e0,0,"
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
