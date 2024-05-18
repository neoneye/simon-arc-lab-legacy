import unittest
from game_of_life import game_of_life, alive_neighbor_counts

class TestGameOfLife(unittest.TestCase):
    def test_empty_grid(self):
        a = ".....,.....,.....,.....,....."
        b = ".....,.....,.....,.....,....."
        self.assertEqual(game_of_life(a), b)
    
    def test_single_cell(self):
        a = ".....,.....,..*..,.....,....."
        b = ".....,.....,.....,.....,....."
        self.assertEqual(game_of_life(a), b)
    
    def test_still_life_block(self):
        """
        https://conwaylife.com/wiki/Block
        """
        a = "....,.**.,.**.,...."
        b = "....,.**.,.**.,...."
        self.assertEqual(game_of_life(a), b)
    
    def test_blinker_1iteration(self):
        """
        https://conwaylife.com/wiki/Blinker
        """
        a = ".....,.....,.***.,.....,....."
        b = ".....,..*..,..*..,..*..,....."
        self.assertEqual(game_of_life(a), b)
    
    def test_blinker_2iterations(self):
        """
        https://conwaylife.com/wiki/Blinker
        """
        a = ".....,.....,.***.,.....,....."
        b = ".....,.....,.***.,.....,....."
        self.assertEqual(game_of_life(a, iterations=2), b)
    
    def test_blinker_3iterations(self):
        """
        https://conwaylife.com/wiki/Blinker
        """
        a = ".....,.....,.***.,.....,....."
        b = ".....,..*..,..*..,..*..,....."
        self.assertEqual(game_of_life(a, iterations=3), b)
    
    def test_oscillator_toad(self):
        """
        https://conwaylife.com/wiki/Toad
        """
        a = "....,.***,***.,...."
        b = "..*.,*..*,*..*,.*.."
        self.assertEqual(game_of_life(a), b)
    
    def test_wrap_x_disabled_by_default(self):
        """
        Verify that there is not wrap around the x-axis
        https://conwaylife.com/wiki/Block
        """
        a = "....,*..*,*..*,...."
        b = "....,....,....,...."
        self.assertEqual(game_of_life(a), b)

    def test_wrap_x_enabled(self):
        """
        Verify that the x-axis wraps around
        https://conwaylife.com/wiki/Block
        """
        a = "....,*..*,*..*,...."
        b = "....,*..*,*..*,...."
        self.assertEqual(game_of_life(a, wrap_x=True), b)

    def test_wrap_y_disabled_by_default(self):
        """
        Verify that there is not wrap around the y-axis
        https://conwaylife.com/wiki/Block
        """
        a = ".**.,....,....,.**."
        b = "....,....,....,...."
        self.assertEqual(game_of_life(a), b)

    def test_alive_neighbor_counts0(self):
        a = "...,..."
        b = [[0, 0, 0], [0, 0, 0]]
        self.assertEqual(alive_neighbor_counts(a), b)

    def test_alive_neighbor_counts1_nowrap(self):
        a = "*..,..."
        b = [[0, 1, 0], [1, 1, 0]]
        self.assertEqual(alive_neighbor_counts(a), b)

    def test_alive_neighbor_counts1_wrapxy(self):
        a = "*...,....,...."
        b = [[0, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1]]
        self.assertEqual(alive_neighbor_counts(a, wrap_x=True, wrap_y=True), b)

    def test_alive_neighbor_counts2(self):
        a = "....,.**.,...."
        b = [[1, 2, 2, 1], [1, 1, 1, 1], [1, 2, 2, 1]]
        self.assertEqual(alive_neighbor_counts(a), b)

if __name__ == '__main__':
    unittest.main()
