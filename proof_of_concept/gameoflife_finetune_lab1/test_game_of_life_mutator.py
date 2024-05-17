from game_of_life_mutator import GameOfLifeMutator
import unittest

class TestGameOfLifeMutator(unittest.TestCase):
    def setUp(self):
        self.mutator = GameOfLifeMutator()

    def test_invalid_row_length(self):
        input = ".**,.**,*.,**.,...,*..,.*."
        with self.assertRaises(ValueError) as context:
            self.mutator.mutate(input)
        self.assertTrue("All rows must have the same length." in str(context.exception))

    def test_invalid_symbols(self):
        input = ".**,.**,.**,**.,...,*..,.*A"
        with self.assertRaises(ValueError) as context:
            self.mutator.mutate(input)
        self.assertTrue("Invalid symbol A found. Only '.' and '*' are allowed." in str(context.exception))

    def test_mutate_underscoredashpipe(self):
        # Arrange
        mutator = GameOfLifeMutator(possible_symbols='_#', row_separators=['|'], pixel_separators=[''])

        # Act
        input = '.**,.*.,**.'
        result = mutator.mutate(input, seed=1)

        # Assert
        self.assertEqual(result['zero_replacement'], '_')
        self.assertEqual(result['one_replacement'], '#')
        self.assertEqual(result['row_separator'], '|')
        self.assertEqual(result['pixel_separator'], '')
        expected = '_##|_#_|##_'
        self.assertEqual(result['mutated_str'], expected)

    def test_mutate_01commanewline(self):
        # Arrange
        mutator = GameOfLifeMutator(possible_symbols='01', row_separators=['\n'], pixel_separators=[', '])

        # Act
        input = '.**,.*.,**.'
        result = mutator.mutate(input, seed=1)

        # Assert
        self.assertEqual(result['zero_replacement'], '0')
        self.assertEqual(result['one_replacement'], '1')
        self.assertEqual(result['row_separator'], '\n')
        self.assertEqual(result['pixel_separator'], ', ')
        expected = '0, 1, 1\n0, 1, 0\n1, 1, 0'
        self.assertEqual(result['mutated_str'], expected)

    def test_mutate_01newline_extraspace(self):
        # Arrange
        mutator = GameOfLifeMutator(possible_symbols='01', row_separators=['\n'], pixel_separators=[''])

        # Act
        input = '.**,.*.,**.'
        result = mutator.mutate(input, num_extra_spaces=20, seed=4)

        # Assert
        self.assertEqual(result['zero_replacement'], '0')
        self.assertEqual(result['one_replacement'], '1')
        self.assertEqual(result['row_separator'], '\n')
        self.assertEqual(result['pixel_separator'], '')
        expected = '  0   1  1     \n0   10\n   11 0 '
        self.assertEqual(result['mutated_str'], expected)

    def test_mutate_01commaspace_extraspace(self):
        # Arrange
        mutator = GameOfLifeMutator(possible_symbols='01', row_separators=[','], pixel_separators=[' '])

        # Act
        input = '.**,.*.,**.'
        result = mutator.mutate(input, num_extra_spaces=2, seed=5)

        # Assert
        self.assertEqual(result['zero_replacement'], '0')
        self.assertEqual(result['one_replacement'], '1')
        self.assertEqual(result['row_separator'], ',')
        self.assertEqual(result['pixel_separator'], ' ')
        expected = '0 1 1,  0 1 0,1   1 0'
        self.assertEqual(result['mutated_str'], expected)

if __name__ == "__main__":
    unittest.main()
