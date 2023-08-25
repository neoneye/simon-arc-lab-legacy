import unittest
from normalize_colors import normalize_color

class TestNormalizeColors(unittest.TestCase):
    def test_normalize_well_defined_range(self):
        from_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        to_colors = []
        for from_color in from_colors:
            color = normalize_color(from_color)
            to_colors.append(color)

        expected = [0, 21, 42, 63, 85, 106, 127, 148, 170, 191, 212, 233, 255]
        self.assertEqual(to_colors, expected)

    def test_normalize_outside_range(self):
        self.assertEqual(normalize_color(-1), 0)
        self.assertEqual(normalize_color(13), 255)
