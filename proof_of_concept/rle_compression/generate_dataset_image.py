# IDEA: transformation "transpose" the image
#
# IDEA: with "rot" prefix, then the image is to be rotated 90 degrees clockwise
#
# IDEA: is there a pixel above, below, left, right, that is the same as the center pixel. All the pixels in the 3x3 area.
# wraparound, wrapx, wrapy, nowrap
#
# IDEA: number of identical neighboring pixels in the 3x3 area. Max 8 pixels can be the same as the center.
# IDEA: number of identical neighboring pixels in the 3x3 area in diagonal corners. Max 4 pixels can be the same as the center.
# IDEA: number of identical neighboring pixels in the 3x3 area in adjacent to center. Max 4 pixels can be the same as the center.
# wraparound, wrapx, wrapy, nowrap
#
# IDEA: multiple size types. corpus: easy, medium, hard
# size10 images 1px to 10px
# size20 images 11px to 20px
# size30 images 21px to 30px
# size40 images 31px to 40px
# size50 images 41px to 50px
#
# IDEA: transformation "rotate" the image
#
# IDEA: transformation "flip" the image
#
# IDEA: transformation "compress x" the image
# IDEA: transformation "compress y" the image
# IDEA: transformation "compress both" the image
#
# IDEA: auto detect what image format it is, and convert it to RLE format.
#
# IDEA: deserialize images with "rot" prefix, then the image is to be rotated 90 degrees clockwise
import json
import os
import random
import numpy as np
from deserialize import deserialize
from serialize import serialize
from image_util import *
from image_create_random_advanced import image_create_random_advanced

def generate_rle_string(seed, max_image_size=100):
    """
    Generate a RLE string of a random image.

    :param seed: The seed for the random number generator
    :param max_image_size: The maximum size of the image
    :return: A tuple of a randomly generated RLE string and the corresponding image
    """

    image = image_create_random_advanced(seed, 1, max_image_size, 1, max_image_size)

    rle_string = serialize(image)
    
    return (rle_string, image)

def generate_serialize_dataset_item(seed):
    """
    Convert from pixel representation to RLE representation.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
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

def generate_deserialize_dataset_item(seed):
    """
    Convert from RLE representation to pixel representation.
    Transform the RLE representation to: histogram, flip, rotate, transpose.

    :param seed: The seed for the random number generator
    :return: A dictionary with the instruction, input, and output
    """
    max_image_size = 10

    instruction_ids = [
        'pixels', 
        'json',
        'histogram',
        'flipx',
        'flipy',
        'transpose',
        'rotate_cw',
        'rotate_ccw',
        'rotate_180',
        'count_same_color_as_center_with_8neighbors_nowrap',
    ]
    instruction_weights = [45, 45, 30, 10, 10, 50, 40, 40, 30, 30]
    instruction_id = random.Random(seed + 1001).choices(instruction_ids, weights=instruction_weights, k=1)[0]

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
    if instruction_id == 'pixels':
        name_output = random.Random(seed + 1002).choice(names_pixels)
    else:
        if instruction_id == 'json':
            name_output = random.Random(seed + 1003).choice(names_json)

    name_inputs = [
        'SIMONARCRLEIMAGE',
        'Simon-ARC-RLE-Image',
        'SimonsRLEImage',
    ]
    name_input = random.Random(seed + 1004).choice(name_inputs)

    instructions_input_output = [
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

    instructions_histogram = [
        f'Histogram of deserialized {name_input}',
        f'histogram of deserialized {name_input}',
        f'Histogram after deserializing {name_input}',
        f'histogram after deserializing {name_input}',
        f'Histogram of {name_input}',
        f'histogram of {name_input}',
        f'Histogram of {name_input}',
        f'convert {name_input} and return the histogram',
        f'Convert {name_input} and return histogram',
        f'Process {name_input} and return the histogram',
        f'process {name_input} and return histogram',
    ]

    instructions_flipx = [
        f'FlipX {name_input}',
        f'Flip-X {name_input}',
        f'flipx {name_input}',
        f'convert {name_input} and return the flipx',
        f'process {name_input} and return flipx',
    ]

    instructions_flipy = [
        f'FlipY {name_input}',
        f'Flip-Y {name_input}',
        f'flipy {name_input}',
        f'convert {name_input} and return the flipy',
        f'process {name_input} and return flipy',
    ]

    instructions_transpose = [
        f'Transpose {name_input}',
        f'transpose {name_input}',
        f'{name_input} transposed',
        f'Process {name_input} and return the transposed',
        f'process {name_input} and return the transposed',
        f'Convert {name_input} and return the transposed',
        f'convert {name_input} and return the transposed',
    ]

    instructions_rotate_cw = [
        f'Rotate Clockwise {name_input}',
        f'Rotate clockwise {name_input}',
        f'Rotate clock-wise {name_input}',
        f'Rotate cw {name_input}',
        f'rotate CW {name_input}',
        f'CW rotate {name_input}',
        f'Process {name_input} and return the clockwise rotated',
        f'process {name_input} and return the cw rotated',
    ]

    instructions_rotate_ccw = [
        f'Rotate CounterClockwise {name_input}',
        f'Rotate counterclockwise {name_input}',
        f'Rotate counter-clock-wise {name_input}',
        f'Rotate ccw {name_input}',
        f'rotate CCW {name_input}',
        f'CCW rotate {name_input}',
        f'Process {name_input} and return the counter clock wise rotated',
        f'process {name_input} and return the ccw rotated',
    ]

    instructions_rotate_180 = [
        f'Rotate 180 {name_input}',
        f'rotate 180 {name_input}',
        f'Half rotate {name_input}',
        f'Half a rotation of {name_input}',
        f'{name_input} rotated halfway',
        f'{name_input} rotated by 180 degrees',
    ]

    instructions_count_same_color_as_center_with_8neighbors_nowrap = [
        f'With {name_input}, 3x3 count neighbors with same color as center',
        f'With {name_input}, Number of neighbors with same color as center',
        f'{name_input}, 3x3 area, how many neighbors have the same color as center',
        f'{name_input}, 3x3 area, count neighbors with same color as center',
    ]

    instructions = instructions_input_output
    if instruction_id == 'histogram':
        instructions = instructions_histogram
    if instruction_id == 'flipx':
        instructions = instructions_flipx
    if instruction_id == 'flipy':
        instructions = instructions_flipy
    if instruction_id == 'transpose':
        instructions = instructions_transpose
    if instruction_id == 'rotate_cw':
        instructions = instructions_rotate_cw
    if instruction_id == 'rotate_ccw':
        instructions = instructions_rotate_ccw
    if instruction_id == 'rotate_180':
        instructions = instructions_rotate_180
    if instruction_id == 'count_same_color_as_center_with_8neighbors_nowrap':
        instructions = instructions_count_same_color_as_center_with_8neighbors_nowrap

    instruction = random.Random(seed + 1005).choice(instructions)

    rle_string, image = generate_rle_string(seed=seed + 1006, max_image_size=max_image_size)

    output = None
    if instruction_id == 'pixels':
        rows = [''.join(map(str, row)) for row in image]
        output = ','.join(rows)
    else:
        if instruction_id == 'json':
            image_list = image.tolist()
            output = json.dumps(image_list, separators=(',', ':'))
        else:
            if instruction_id == 'histogram':
                output = pretty_histogram_of_image(image)
            else:
                if instruction_id == 'flipx':
                    flipped_image = image[:, ::-1]
                    output_rle_string = serialize(flipped_image)
                    output = output_rle_string
                else:
                    if instruction_id == 'flipy':
                        flipped_image = image[::-1, :]
                        output_rle_string = serialize(flipped_image)
                        output = output_rle_string
                    else:
                        if instruction_id == 'transpose':
                            transposed_image = image.transpose()
                            output_rle_string = serialize(transposed_image)
                            output = output_rle_string
                        else:
                            if instruction_id == 'rotate_cw':
                                new_image = image_rotate_cw(image)
                                output_rle_string = serialize(new_image)
                                output = output_rle_string
                            else:
                                if instruction_id == 'rotate_ccw':
                                    new_image = image_rotate_ccw(image)
                                    output_rle_string = serialize(new_image)
                                    output = output_rle_string
                                else:
                                    if instruction_id == 'rotate_180':
                                        new_image = image_rotate_180(image)
                                        output_rle_string = serialize(new_image)
                                        output = output_rle_string
                                    else:
                                        if instruction_id == 'count_same_color_as_center_with_8neighbors_nowrap':
                                            new_image = count_same_color_as_center_with_8neighbors_nowrap(image)
                                            output_rle_string = serialize(new_image)
                                            output = output_rle_string
                                        else:
                                            raise Exception("Unreachable code reached")

    dict = {
        'instruction': instruction,
        'input': rle_string,
        'output': output
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=400000):
    dataset = []
    dataset_byte_size = 0
    for i in range(max_num_samples):
        if i % 2 == 0:
            item = generate_serialize_dataset_item(seed_start + i)
        else:
            item = generate_deserialize_dataset_item(seed_start + i)
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
filename = 'data_image.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

