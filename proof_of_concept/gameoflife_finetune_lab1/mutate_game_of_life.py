import random

def mutate_game_of_life(game_of_life_str, seed=None):
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
    
    grid = [[1 if cell == '*' else 0 for cell in row] for row in rows]
    
    # Step 2: Convert JSON-like structure to a string representation
    # Define possible replacement symbols
    possible_symbols  = 'abcdefghijklmnopqrstuvwxyz'
    possible_symbols += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    possible_symbols += '0123456789'
    possible_symbols += ',|_-!@#$%^&*().@#?'
    possible_symbols += 'αβγδεζηθικλμνξοπρστυφχψω'
    possible_symbols += 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
    possible_symbols += '🌱🌿🌻🌟✨💫🔥⚡🌕🌑⭐☀️🌈💀☠️👻🎃🦠'
    possible_symbols += '░▒▓█'
    
    # Select random symbols for 1 and 0
    one_replacement = random.choice(possible_symbols)
    zero_replacement = random.choice(possible_symbols)
    
    # Ensure replacements are different
    while one_replacement == zero_replacement:
        zero_replacement = random.choice(possible_symbols)
    
    # Define possible separators
    row_separators = ['\n', '|', ';', ',']
    pixel_separators = ['', ',', ' ', ', ', '|']
    
    # Filter out symbols used for 1 and 0 from the separators
    filtered_row_separators = [sep for sep in row_separators if sep not in [one_replacement, zero_replacement]]
    filtered_pixel_separators = [sep for sep in pixel_separators if sep not in [one_replacement, zero_replacement]]
    
    # Select random separators for rows and pixels
    row_sep = random.choice(filtered_row_separators)
    pixel_sep = random.choice(filtered_pixel_separators)

    if row_sep == ',' and pixel_sep == ',':
        pixel_sep = ''

    if row_sep == ',' and pixel_sep == ', ':
        pixel_sep = ''
    
    # Convert the grid to a string with the new separators and symbols
    def convert_cell(cell):
        return one_replacement if cell == 1 else zero_replacement
    
    converted_rows = [pixel_sep.join(convert_cell(cell) for cell in row) for row in grid]
    converted_str = row_sep.join(converted_rows)
    
    return converted_str, one_replacement, zero_replacement, row_sep, pixel_sep

# Example usage
game_of_life_str = ".**,.**,.**,**.,...,*..,.*."
mutated_str, one_replacement, zero_replacement, row_sep, pixel_sep = mutate_game_of_life(game_of_life_str, seed=46)

print("Original:", game_of_life_str)
print("Mutated:", mutated_str)
print("1 replaced with:", one_replacement)
print("0 replaced with:", zero_replacement)
print("Row separator:", repr(row_sep))
print("Pixel separator:", repr(pixel_sep))
