import numpy as np
import random

def image_create(width, height, color):
    image = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = color
    return image


def image_create_random_with_two_colors(width, height, color1, color2, ratio, seed):
    image = image_create(width, height, color1)

    positions = []
    for y in range(height):
        for x in range(width):
            positions += [(y, x)]

    random.Random(seed).shuffle(positions)

    # take a ratio of the positions
    num_positions = int(len(positions) * ratio)
    for i in range(num_positions):
        y, x = positions[i]
        image[y, x] = color2
    return image

