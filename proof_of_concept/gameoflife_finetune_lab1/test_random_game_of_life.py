import unittest
from random_game_of_life import generate_random_game_of_life_string

class TestRandomGameOfLife(unittest.TestCase):
    def test_size5x5(self):
        actual = generate_random_game_of_life_string(seed=0, min_width=5, max_width=5, min_height=5, max_height=5)
        expected = ".****,**..*,..*.*,..**.,***.*"
        self.assertEqual(actual, expected)
    
    def test_size6x3(self):
        actual = generate_random_game_of_life_string(seed=0, min_width=6, max_width=6, min_height=3, max_height=3)
        expected = ".*****,*..*..,*.*..*"
        self.assertEqual(actual, expected)
    
    def test_size6x3(self):
        actual = generate_random_game_of_life_string(seed=0, min_width=6, max_width=6, min_height=3, max_height=3)
        expected = ".*****,*..*..,*.*..*"
        self.assertEqual(actual, expected)
    
    def test_variable_size(self):
        min_size = 5
        max_size = 10
        output = generate_random_game_of_life_string(seed=0, min_width=min_size, max_width=max_size, min_height=min_size, max_height=max_size)
        width = len(output.split(',')[0])
        height = len(output.split(','))
        self.assertGreaterEqual(width, min_size)
        self.assertLessEqual(width, max_size)
        self.assertGreaterEqual(height, min_size)
        self.assertLessEqual(height, max_size)
    
if __name__ == '__main__':
    unittest.main()
