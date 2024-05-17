import random

def generate_random_game_of_life_string(seed=None, min_width=3, max_width=10, min_height=3, max_height=10):
    # Set the random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Generate random width and height between min and max
    width = random.randint(min_width, max_width)
    height = random.randint(min_height, max_height)
    
    # Generate the grid
    grid = []
    for _ in range(height):
        row = ''.join(random.choice(['.', '*']) for _ in range(width))
        grid.append(row)
    
    # Convert the grid to the required string format
    return ','.join(grid)

if __name__ == "__main__":
    random_grid_string = generate_random_game_of_life_string(seed=42)
    print(random_grid_string)
