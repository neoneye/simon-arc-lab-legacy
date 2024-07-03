import numpy as np

def serialize(image):
    height, width = image.shape
    s = f"{width} {height} "
    last_line = ""
    for y in range(height):
        if y > 0:
            s += ','
        color = image[y, 0]
        is_same_color = np.all(image[y, :] == color)
        if is_same_color:
            current_line = str(color)
            if current_line != last_line:
                s += current_line
                last_line = current_line
            continue
        color = image[y, 0]
        count = 1
        current_line = ""
        for x in range(1, width):
            new_color = image[y, x]
            if count < 25 and new_color == color:
                count += 1
                continue
            if count >= 2:
                current_line += chr((count - 2) + ord('a'))
            current_line += str(color)
            color = new_color
            count = 1
        if count >= 2:
            current_line += chr((count - 2) + ord('a'))
            current_line += str(color)
        else:
            current_line += str(color)
        if len(current_line) > 1 and current_line == last_line:
            # When the string is empty, it means that the line is the same as the previous line.
            continue
        else:
            s += current_line
            last_line = current_line

    # verify_parse = False
    # if verify_parse:
    #     parsed = parse_runlengthencoded_variant1(s)
    #     if not np.array_equal(parsed, image):
    #         print(f"parsed: {parsed}")
    #         print(f"image: {image}")
    #         raise ValueError("parsed image is not the same as the original image")

    return s

# Example usage
# image = np.array([
#     [0, 0, 2, 2, 0],
#     [0, 0, 2, 2, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 2, 2, 0],
#     [0, 0, 0, 0, 0],
# ], dtype=np.uint8)

# rle_string = format_image_runlengthencoded_variant1(image)
# print(rle_string)
