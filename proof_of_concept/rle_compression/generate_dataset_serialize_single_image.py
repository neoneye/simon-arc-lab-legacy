# IDEA: multiple size types. corpus: easy, medium, hard
# size10 images 1px to 10px
# size20 images 11px to 20px
# size30 images 21px to 30px
# size40 images 31px to 40px
# size50 images 41px to 50px
#
# IDEA: multiple randomizers.
# 5 or more color images.
#
# IDEA: with "rot" prefix, then the image is to be rotated 90 degrees clockwise
#
# IDEA: transformation "rotate" the image
#
# IDEA: transformation "flip" the image
#
# IDEA: transformation "transpose" the image
#
# IDEA: auto detect what image format it is, and convert it to RLE format.
import json
import os
import random
import numpy as np
from deserialize import deserialize
from serialize import serialize
from image_util import *

def generate_rle_string(seed, max_image_size=100):
    """
    Generate a RLE string of a random image.

    :param seed: The seed for the random number generator
    :param max_image_size: The maximum size of the image
    :return: A tuple of a randomly generated RLE string and the corresponding image
    """

    width = random.Random(seed + 1).randint(1, max_image_size)
    height = random.Random(seed + 2).randint(1, max_image_size)

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.Random(seed + 3).shuffle(colors)
    color0 = colors[0]
    color1 = colors[1]
    color2 = colors[2]
    color3 = colors[3]

    image_types = ['two_colors', 'three_colors', 'four_colors']
    image_type = random.Random(seed + 4).choice(image_types)

    image = None
    if image_type == 'two_colors':
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        ratio = random.Random(seed + 5).choice(ratios)
        image = image_create_random_with_two_colors(width, height, color0, color1, ratio, seed + 6)
    if image_type == 'three_colors':
        weights = [1, 1, 1, 2, 3, 4, 7, 11]
        random.Random(seed + 5).shuffle(weights)
        weight0 = weights[0]
        weight1 = weights[1]
        weight2 = weights[2]
        image = image_create_random_with_three_colors(width, height, color0, color1, color2, weight0, weight1, weight2, seed + 10)
    if image_type == 'four_colors':
        weights = [1, 1, 1, 1, 2, 2, 3, 3, 4, 7, 11]
        random.Random(seed + 5).shuffle(weights)
        weight0 = weights[0]
        weight1 = weights[1]
        weight2 = weights[2]
        weight3 = weights[3]
        image = image_create_random_with_four_colors(width, height, color0, color1, color2, color3, weight0, weight1, weight2, weight3, seed + 10)

    rle_string = serialize(image)
    return (rle_string, image)

def generate_dataset_item(seed):
    max_image_size = 10

    input_formats = [
        'pixels', 
        'json'
    ]
    input_format_weights = [45, 45]
    input_format = random.Random(seed + 1001).choices(input_formats, weights=input_format_weights, k=1)[0]

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

    name_input = None
    if input_format == 'pixels':
        name_input = random.Random(seed + 1002).choice(names_pixels)
    else:
        if input_format == 'json':
            name_input = random.Random(seed + 1003).choice(names_json)

    name_outputs = [
        'SIMONARCRLEIMAGE',
        'Simon-ARC-RLE-Image',
        'SimonsRLEImage',
    ]
    name_output = random.Random(seed + 1004).choice(name_outputs)

    instructions_input_output = [
        f'Serialize {name_input} to {name_output}',
        f'serialize {name_input} to {name_output}',
        f'convert {name_input} to {name_output}',
        f'Convert {name_input} to {name_output}',
        f'Transform {name_input} to {name_output}',
        f'transform {name_input} to {name_output}',
        f'Change {name_input} to {name_output}',
        f'change {name_input} to {name_output}',
        f'{name_input} to {name_output}',
        f'{name_output} from {name_input}',
    ]

    instructions = instructions_input_output

    instruction = random.Random(seed + 1005).choice(instructions)

    rle_string, image = generate_rle_string(seed=seed + 1006, max_image_size=max_image_size)

    output = rle_string

    input = None
    if input_format == 'pixels':
        rows = [''.join(map(str, row)) for row in image]
        input = ','.join(rows)
    else:
        if input_format == 'json':
            image_list = image.tolist()
            input = json.dumps(image_list, separators=(',', ':'))

    dict = {
        'instruction': instruction,
        'input': input,
        'output': output
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=400000):
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
filename = 'data_serialize_single_image.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

