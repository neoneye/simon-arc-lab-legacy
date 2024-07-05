# IDEA: multiple randomizers.
# two-color images, halfnhalf of two colors, or 2/3 of one color and 1/3 of another color.
# three-color images, 1/3 of each color, or 1/2 of one color and 1/4 of each of the other two colors.
#
# IDEA: multiple size types.
# size10 images 1px to 10px
# size20 images 11px to 20px
# size30 images 21px to 30px
# size40 images 31px to 40px
# size50 images 41px to 50px
#
# IDEA: with "rot" prefix, then the image is to be rotated 90 degrees clockwise
#
# IDEA: histogram of what colors are present, and the frequency of each color
#
import json
import os
import random
import numpy as np
from deserialize import deserialize
from serialize import serialize

def generate_rle_string(max_image_size=100, seed=None):
    """
    Generate a RLE string of a random image.

    :param max_image_size: The maximum size of the image
    :param seed: The seed for the random number generator
    :return: A tuple of a randomly generated RLE string and the corresponding image
    """
    if seed is not None:
        random.seed(seed)

    width = random.randint(1, max_image_size)
    height = random.randint(1, max_image_size)

    available_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    background_color = random.choice(available_colors)
    available_colors.remove(background_color)
    random.shuffle(available_colors)

    image = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = background_color


    positions = []
    for y in range(height):
        for x in range(width):
            positions += [(y, x)]

    random.shuffle(positions)

    color = random.choice(available_colors)

    # take half of the positions
    num_positions = len(positions) // 2
    for i in range(num_positions):
        y, x = positions[i]
        image[y, x] = color

    rle_string = serialize(image)
    return (rle_string, image)

def generate_dataset_item(seed):
    max_image_size = 10

    output_formats = [
        'pixels', 
        'json'
    ]
    output_format_weights = [0.45, 0.45]
    output_format = random.Random(seed + 1001).choices(output_formats, weights=output_format_weights, k=1)[0]

    names_pixels = [
        'Pixels',
        'pixels',
        'Digits',
        'digits',
        'Symbols',
        'symbols',
        'String',
        'string',
    ]
    names_json = [
        'Json',
        'json',
        'JSON',
    ]

    name_output = None
    if output_format == 'pixels':
        name_output = random.Random(seed + 1002).choice(names_pixels)
    else:
        if output_format == 'json':
            name_output = random.Random(seed + 1003).choice(names_json)

    name_inputs = [
        'SIMONARCRLEIMAGE',
        'Simon-ARC-RLE-Image',
        'SimonsRLEImage',
    ]
    name_input = random.Random(seed + 1004).choice(name_inputs)

    instructions = [
        f'Deserialize {name_input} to {name_output}',
        f'deserialize {name_input} to {name_output}',
        f'convert {name_input} to {name_output}',
        f'Convert {name_input} to {name_output}',
        f'Transform {name_input} to {name_output}',
        f'transform {name_input} to {name_output}',
        f'Change {name_input} to {name_output}',
        f'change {name_input} to {name_output}',
        f'{name_input} to {name_output}',
        f'{name_output} from {name_input}',
    ]

    instruction = random.Random(seed + 1005).choice(instructions)

    rle_string, image = generate_rle_string(max_image_size=max_image_size, seed=seed + 1006)

    output = None
    if output_format == 'pixels':
        rows = [''.join(map(str, row)) for row in image]
        output = ','.join(rows)
    else:
        if output_format == 'json':
            image_list = image.tolist()
            output = json.dumps(image_list, separators=(',', ':'))

    dict = {
        'instruction': instruction,
        'input': rle_string,
        'output': output
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=300000):
    dataset = []
    dataset_byte_size = 0
    for i in range(max_num_samples):
        item = generate_dataset_item(seed_start + i)
        bytes = len(json.dumps(item))
        if dataset_byte_size + bytes > max_byte_size:
            break
        dataset_byte_size += bytes
        dataset.append(item)
    return dataset

dataset = generate_dataset(
    max_num_samples=50,
    max_byte_size=1024*1024*20,
)

# Save dataset to file
filename = 'data_deserialize_single_image.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

