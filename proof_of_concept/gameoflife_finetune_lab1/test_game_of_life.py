import unittest
from game_of_life import GameOfLife

def game_of_life(input_str, wrap_x=False, wrap_y=False, iterations=1):
    return GameOfLife.create(input_str, wrap_x=wrap_x, wrap_y=wrap_y, iterations=iterations)

class TestGameOfLife(unittest.TestCase):
    def test_empty_grid(self):
        a = ".....,.....,.....,.....,....."
        b = ".....,.....,.....,.....,....."
        gol = game_of_life(a)
        self.assertEqual(gol.output_str, b)
    
    def test_single_cell(self):
        a = ".....,.....,..*..,.....,....."
        b = ".....,.....,.....,.....,....."
        gol = game_of_life(a)
        self.assertEqual(gol.output_str, b)
    
    def test_still_life_block(self):
        """
        https://conwaylife.com/wiki/Block
        """
        a = "....,.**.,.**.,...."
        b = "....,.**.,.**.,...."
        gol = game_of_life(a)
        self.assertEqual(gol.output_str, b)
    
    def test_blinker_1iteration(self):
        """
        https://conwaylife.com/wiki/Blinker
        """
        a = ".....,.....,.***.,.....,....."
        b = ".....,..*..,..*..,..*..,....."
        gol = game_of_life(a)
        self.assertEqual(gol.output_str, b)
    
    def test_blinker_2iterations(self):
        """
        https://conwaylife.com/wiki/Blinker
        """
        a = ".....,.....,.***.,.....,....."
        b = ".....,.....,.***.,.....,....."
        gol = game_of_life(a, iterations=2)
        self.assertEqual(gol.output_str, b)
    
    def test_blinker_3iterations(self):
        """
        https://conwaylife.com/wiki/Blinker
        """
        a = ".....,.....,.***.,.....,....."
        b = ".....,..*..,..*..,..*..,....."
        gol = game_of_life(a, iterations=3)
        self.assertEqual(gol.output_str, b)
    
    def test_oscillator_toad(self):
        """
        https://conwaylife.com/wiki/Toad
        """
        a = "....,.***,***.,...."
        b = "..*.,*..*,*..*,.*.."
        gol = game_of_life(a)
        self.assertEqual(gol.output_str, b)
    
    def test_wrap_x_disabled_by_default(self):
        """
        Verify that there is not wrap around the x-axis
        https://conwaylife.com/wiki/Block
        """
        a = "....,*..*,*..*,...."
        b = "....,....,....,...."
        gol = game_of_life(a)
        self.assertEqual(gol.output_str, b)

    def test_wrap_x_enabled(self):
        """
        Verify that the x-axis wraps around
        https://conwaylife.com/wiki/Block
        """
        a = "....,*..*,*..*,...."
        b = "....,*..*,*..*,...."
        gol = game_of_life(a, wrap_x=True)
        self.assertEqual(gol.output_str, b)

    def test_wrap_y_disabled_by_default(self):
        """
        Verify that there is not wrap around the y-axis
        https://conwaylife.com/wiki/Block
        """
        a = ".**.,....,....,.**."
        b = "....,....,....,...."
        gol = game_of_life(a)
        self.assertEqual(gol.output_str, b)

    def test_alive_neighbor_counts0(self):
        a = "...,..."
        b = [[0, 0, 0], [0, 0, 0]]
        gol = game_of_life(a)
        self.assertEqual(gol.alive_neighbor_counts, b)

    def test_alive_neighbor_counts1_nowrap(self):
        a = "*..,..."
        b = [[0, 1, 0], [1, 1, 0]]
        gol = game_of_life(a)
        self.assertEqual(gol.alive_neighbor_counts, b)

    def test_alive_neighbor_counts1_wrapxy(self):
        a = "*...,....,...."
        b = [[0, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1]]
        gol = game_of_life(a, wrap_x=True, wrap_y=True)
        self.assertEqual(gol.alive_neighbor_counts, b)

    def test_alive_neighbor_counts2(self):
        a = "....,.**.,...."
        b = [[1, 2, 2, 1], [1, 1, 1, 1], [1, 2, 2, 1]]
        gol = game_of_life(a)
        self.assertEqual(gol.alive_neighbor_counts, b)

if __name__ == '__main__':
    unittest.main()
