import unittest
from list_util import list_compress

class TestListUtil(unittest.TestCase):
    def test_list_compress(self):
        input = [2, 2, 3, 3, 5, 3, 5, 5, 5, 2, 2, 1, 0, 1, 0, 0]
        expected = [2, 3, 5, 3, 5, 2, 1, 0, 1, 0]
        actual = list_compress(input)
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
