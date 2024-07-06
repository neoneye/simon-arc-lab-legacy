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

def image_create_random_with_three_colors(width, height, color0, color1, color2, weight0, weight1, weight2, seed):
    image = image_create(width, height, color0)

    positions = [(y, x) for y in range(height) for x in range(width)]

    random.Random(seed).shuffle(positions)

    total_weight = weight0 + weight1 + weight2
    num_positions_a = int(len(positions) * (weight1 / total_weight))
    num_positions_b = int(len(positions) * (weight2 / total_weight))

    for i in range(num_positions_a):
        y, x = positions[i]
        image[y, x] = color1

    for i in range(num_positions_b):
        y, x = positions[i + num_positions_a]
        image[y, x] = color2

    return image
