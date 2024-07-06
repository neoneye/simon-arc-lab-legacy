import numpy as np

def image_create(width, height, color):
    image = np.zeros((height, width), dtype=np.uint8)
    image_fill(image, color)
    return image


def image_fill(image, color):
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            image[y, x] = color


