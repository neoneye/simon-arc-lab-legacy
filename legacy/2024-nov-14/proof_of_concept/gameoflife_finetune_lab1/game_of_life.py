class GameOfLife:
    def __init__(self, input_str, wrap_x, wrap_y):
        self.input_str = input_str
        self.wrap_x = wrap_x
        self.wrap_y = wrap_y
        self.grid = GameOfLife.parse_input_str(input_str)
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.alive_neighbor_counts = GameOfLife.alive_neighbor_counts(self.width, self.height, self.grid, wrap_x, wrap_y)
        self.output_str = GameOfLife.compute_game_of_life(self.width, self.height, self.grid, self.alive_neighbor_counts)

    @classmethod
    def create(cls, input_str, wrap_x, wrap_y, iterations):
        state = GameOfLife(input_str, wrap_x=wrap_x, wrap_y=wrap_y)
        for _ in range(iterations-1):
            state = state.next_state()
        return state

    @classmethod
    def parse_input_str(cls, input_str):
        rows = input_str.split(',')
        return [list(row) for row in rows]

    @classmethod
    def count_neighbors(cls, width, height, grid, x, y, wrap_x, wrap_y):
        """
        Count the number of alive neighbors for one cell.
        """
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

    @classmethod
    def alive_neighbor_counts(cls, width, height, grid, wrap_x, wrap_y):
        """
        Counts the number of alive neighbors for every cell in the entire grid.
        """
        counts = [[GameOfLife.count_neighbors(width, height, grid, x, y, wrap_x, wrap_y) for x in range(width)] for y in range(height)]
        return counts

    @classmethod
    def compute_game_of_life(cls, width, height, grid, counts):
        new_grid = [['.' for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                alive_neighbors = counts[y][x]
                if grid[y][x] == '*' and (alive_neighbors == 2 or alive_neighbors == 3):
                    new_grid[y][x] = '*'
                elif grid[y][x] == '.' and alive_neighbors == 3:
                    new_grid[y][x] = '*'
                else:
                    new_grid[y][x] = '.'

        new_rows = [''.join(row) for row in new_grid]
        return ','.join(new_rows)
    
    def next_state(self):
        return GameOfLife(self.output_str, wrap_x=self.wrap_x, wrap_y=self.wrap_y)
