import random
import string
from deserialize import deserialize, decode_rle_row

def generate_rle_string(length=10):
    """
    Generate a random RLE string of the specified length.

    :param length: The desired length of the RLE string
    :return: A randomly generated RLE string
    """
    rle_string = ''
    while len(rle_string) < length:
        digit = str(random.randint(0, 9))
        alpha = random.choice(string.ascii_lowercase)
        run_length = random.randint(1, 9)

        # If the run length is greater than 3, add the alpha character
        if run_length > 3:
            rle_string += digit + alpha
        else:
            rle_string += digit

        # Ensure we don't exceed the desired length
        if len(rle_string) > length:
            rle_string = rle_string[:length]
            break

    return rle_string

# Generate a set of example RLE strings
for _ in range(10):
    s = generate_rle_string(10)
    print(s)
    # pixels = decode_rle_row(s, 10)
    # print(pixels)

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
