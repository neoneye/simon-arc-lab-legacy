import random

class GameOfLifeMutator:
    def __init__(self, possible_symbols=None, row_separators=None, pixel_separators=None):
        if possible_symbols is None:
            possible_symbols  = 'abcdefghijklmnopqrstuvwxyz'
            possible_symbols += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            possible_symbols += '0123456789'
            possible_symbols += '_-!@#$%^&*()'
            possible_symbols += 'αβγδεζηθικλμνξοπρστυφχψω'
            possible_symbols += 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
            possible_symbols += '🌱🌿🌻🌟✨💫🔥⚡🌕🌑⭐☀️🌈💀☠️👻🎃🦠'
            possible_symbols += '░▒▓█'
        self.possible_symbols = possible_symbols
        
        if row_separators is None:
            row_separators = ['\n', '|', ';', ',']
        self.row_separators = row_separators
        
        if pixel_separators is None:
            pixel_separators = ['', ',', ' ', ', ', '|']
        self.pixel_separators = pixel_separators
    
    def mutate(self, game_of_life_str, num_extra_spaces=0, seed=None):
        """
        The "game_of_life_str" looks like this '.**,.*.,**.'
        '.' represents a dead cell.
        '*' represents a live cell.
        ',' separates the rows.

        The "num_extra_spaces=5" parameter adds 5 extra spaces at random positions,
        having some defects in the data, makes the finetuned model more robust to defects.
        """
        # Set the random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
        
        # Step 1: Convert game_of_life_str to JSON-like list of lists
        rows = game_of_life_str.split(',')
        
        # Check for consistent row lengths and valid symbols
        row_length = len(rows[0])
        valid_symbols = {'.', '*'}
        for row in rows:
            if len(row) != row_length:
                raise ValueError(f"All rows must have the same length. Expected {row_length}, but got {len(row)}.")
            for cell in row:
                if cell not in valid_symbols:
                    raise ValueError(f"Invalid symbol {cell} found. Only '.' and '*' are allowed.")
        
        # Step 2: Insert extra spaces into the game_of_life_str
        game_of_life_list = list(game_of_life_str)
        for _ in range(num_extra_spaces):
            pos = random.randint(0, len(game_of_life_list))
            game_of_life_list.insert(pos, ' ')
        game_of_life_str = ''.join(game_of_life_list)
        
        # Step 3: Convert the updated game_of_life_str to a grid
        # 0 represents a dead cell, 1 represents live cell, 2 represents the extra space.
        rows = game_of_life_str.split(',')
        grid = []
        for row in rows:
            grid_row = []
            for cell in row:
                if cell == '*':
                    grid_row.append(1)
                elif cell == ' ':
                    grid_row.append(2)
                else:
                    grid_row.append(0)
            grid.append(grid_row)
        
        # Step 4: Convert JSON-like structure to a string representation
        # Select random symbols for 1, 0, and 2 (extra spaces)
        one_replacement = random.choice(self.possible_symbols)
        zero_replacement = random.choice(self.possible_symbols)
        space = ' '
        
        # Ensure replacements are different
        while one_replacement == zero_replacement:
            one_replacement = random.choice(self.possible_symbols)
            zero_replacement = random.choice(self.possible_symbols)
        
        # Filter out symbols used for 1, 0, and 2 from the separators
        ignore_symbols = [one_replacement, zero_replacement]
        filtered_row_separators = [sep for sep in self.row_separators if sep not in ignore_symbols]
        filtered_pixel_separators = [sep for sep in self.pixel_separators if sep not in ignore_symbols]
        
        # Select random separators for rows and pixels
        row_sep = random.choice(filtered_row_separators)
        pixel_sep = random.choice(filtered_pixel_separators)

        if row_sep == ',' and pixel_sep == ',':
            pixel_sep = ''
        if row_sep == ',' and pixel_sep == ', ':
            pixel_sep = ''
        
        # Convert the grid to a string with the new separators and symbols
        def convert_cell(cell):
            if cell == 1:
                return one_replacement
            elif cell == 0:
                return zero_replacement
            elif cell == 2:
                return space
        
        converted_rows = [pixel_sep.join(convert_cell(cell) for cell in row) for row in grid]
        converted_str = row_sep.join(converted_rows)
        
        return {
            'mutated_str': converted_str,
            'one_replacement': one_replacement,
            'zero_replacement': zero_replacement,
            'row_separator': row_sep,
            'pixel_separator': pixel_sep
        }

if __name__ == "__main__":
    mutator = GameOfLifeMutator()
    game_of_life_str = '.**,.*.,**.'
    mutated = mutator.mutate(game_of_life_str, seed=42, num_extra_spaces=5)
    print(mutated)
