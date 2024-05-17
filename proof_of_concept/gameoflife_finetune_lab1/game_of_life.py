def game_of_life(input_str):
    rows = input_str.split(',')
    grid = [list(row) for row in rows]
    
    def count_neighbors(x, y):
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),         (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                if grid[nx][ny] == '*':
                    count += 1
        return count

    new_grid = [['.' for _ in range(len(grid[0]))] for _ in range(len(grid))]

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            alive_neighbors = count_neighbors(i, j)
            if grid[i][j] == '*' and (alive_neighbors == 2 or alive_neighbors == 3):
                new_grid[i][j] = '*'
            elif grid[i][j] == '.' and alive_neighbors == 3:
                new_grid[i][j] = '*'
            else:
                new_grid[i][j] = '.'

    new_rows = [''.join(row) for row in new_grid]
    return ','.join(new_rows)
