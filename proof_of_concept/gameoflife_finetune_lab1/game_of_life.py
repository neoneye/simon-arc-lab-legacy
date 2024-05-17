def game_of_life(input_str, wrap_x=False, wrap_y=False):
    rows = input_str.split(',')
    grid = [list(row) for row in rows]
    height = len(grid)
    width = len(grid[0])
    
    def count_neighbors(x, y):
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),         (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        count = 0
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            
            if wrap_x:
                nx = (nx + width) % width
            if wrap_y:
                ny = (ny + height) % height
            
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny][nx] == '*':
                    count += 1
        return count

    new_grid = [['.' for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            alive_neighbors = count_neighbors(x, y)
            if grid[y][x] == '*' and (alive_neighbors == 2 or alive_neighbors == 3):
                new_grid[y][x] = '*'
            elif grid[y][x] == '.' and alive_neighbors == 3:
                new_grid[y][x] = '*'
            else:
                new_grid[y][x] = '.'

    new_rows = [''.join(row) for row in new_grid]
    return ','.join(new_rows)