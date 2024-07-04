import random
from deserialize import deserialize, decode_rle_row, decode_rle_row_inner

def generate_rle_string(string_length=10, pixel_length=50):
    """
    Generate a random RLE string of the specified length.

    :param length: The desired length of the RLE string
    :return: A randomly generated RLE string
    """
    rle_string = ''
    current_pixel_length = 0
    current_pixels = []
    while len(rle_string) < string_length and current_pixel_length < pixel_length:
        digit = str(random.randint(0, 9))
        run_length = random.randint(1, 27)

        if run_length > 1:
            alpha_char = chr(ord('a') + (run_length - 2))
            rle_string += alpha_char + digit
        else:
            rle_string += digit

        current_pixels, current_pixel_length = decode_rle_row_inner(rle_string)

    return (rle_string, current_pixels, current_pixel_length)

# Generate a set of example RLE strings
for _ in range(10):
    rle_string, current_pixels, pixel_length = generate_rle_string(10)
    print(rle_string, current_pixels, pixel_length)

# def generate_rle_dataset(num_samples, width, height):
#     dataset = []
#     for _ in range(num_samples):
#         # Generate random RLE strings and decode them
#         rle_strings = []
#         for _ in range(height):
#             rle_row = generate_rle_string(width)  # Assuming generate_rle_string generates valid RLE strings for a row
#             decoded_row = decode_rle_row(rle_row, width)
#             rle_strings.append(rle_row)
#         rle_string = f"{width} {height} " + ','.join(rle_strings)
#         image = parse_runlengthencoded_variant1(rle_string)
#         dataset.append((rle_string, image))
#     return dataset

# # Example usage of dataset generator
# dataset = generate_rle_dataset(5, 11, 11)
# for rle_string, image in dataset:
#     print(rle_string)
#     print(image)
