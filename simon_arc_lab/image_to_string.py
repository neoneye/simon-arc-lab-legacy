import numpy as np

def image_to_string(image: np.array) -> str:
    height, _ = image.shape
    rows = []
    for y in range(height):
        pixels = image[y, :]
        s = "".join([str(pixel) for pixel in pixels])
        rows.append(s)
    return "\n".join(rows)
