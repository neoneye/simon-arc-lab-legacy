import unittest
from random_data import generate_random_byte_array

class TestRandomData(unittest.TestCase):
    def test_generate_random_byte_array_empty(self):
        actual = generate_random_byte_array(length=0, seed=0)
        expected = bytearray()
        self.assertEqual(actual, expected)

    def test_generate_random_byte_array_length1(self):
        actual = generate_random_byte_array(length=1, seed=0)
        expected = bytearray([0xc5])
        self.assertEqual(actual, expected)

    def test_generate_random_byte_array_length2(self):
        actual = generate_random_byte_array(length=2, seed=0)
        expected = bytearray([0xc5, 0xd7])
        self.assertEqual(actual, expected)
    
if __name__ == '__main__':
    unittest.main()
