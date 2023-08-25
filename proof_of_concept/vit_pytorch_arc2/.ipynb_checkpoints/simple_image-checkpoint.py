import numpy as np

def image_new(width: int, height: int, color: int):
    rows = []
    for y in range(height):
        columns = [color] * width
        rows.append(columns)
    pixels = np.array(rows, np.int32)
    return pixels

def set_pixel(pixels: np.ndarray, x: int, y: int, color: int):
    if x < 0 or y < 0:
        return
    shape = pixels.shape
    if x >= shape[1] or y >= shape[0]:
        return
    pixels[y][x] = color
    
def draw_rect(pixels: np.ndarray, x: int, y: int, width: int, height: int, color: int):
    for i in range(width):
        for j in range(height):
            set_pixel(pixels, x + i, y + j, color)

def draw_box(pixels: np.ndarray, x: int, y: int, width: int, height: int, color: int):
    y1 = y + height - 1
    for i in range(width):
        set_pixel(pixels, x + i, y, color)
        set_pixel(pixels, x + i, y1, color)
    x1 = x + width - 1
    for j in range(height):
        set_pixel(pixels, x, y + j, color)
        set_pixel(pixels, x1, y + j, color)

if __name__ == "__main__":
    pixels = image_new(5, 7, 0)
    print(pixels.shape)
    draw_rect(pixels, 1, 1, 3, 2, 1)
    draw_box(pixels, 0, 0, 5, 7, 2)
    set_pixel(pixels, 2, 3, 3)
    print(pixels)
    