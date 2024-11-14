import numpy as np
from normalize_colors import normalize_color
from PIL import Image

def pil_image_from_pixels_with_red_palette(pixels: np.ndarray) -> Image:
    width = pixels.shape[1]
    height = pixels.shape[0]
    pil_image = Image.new('RGB', (width, height))
    for row_index, rows in enumerate(pixels):
        for column_index, pixel in enumerate(rows):
            color_red = normalize_color(pixel)
            color_green = 0
            color_blue = 0
            pil_image.putpixel((column_index,row_index), (color_red,color_green,color_blue))
    return pil_image

if __name__ == "__main__":
    raw = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 255]
    pixels = np.array([raw], np.int32)
    image = pil_image_from_pixels_with_red_palette(pixels)
    image.show()
