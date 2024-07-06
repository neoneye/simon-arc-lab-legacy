# IDEA: compress row, by removing a-z length indicator, and remove duplicate colors adjacent, so it's only the unique pixel colors.
import json
import os
import random
from image_util import pretty_histogram_of_image, image_create
from deserialize import decode_rle_row_inner
from serialize import rle_serialize_line_inner

def generate_rle_string(string_length=10, pixel_length=50, seed=None):
    """
    Generate a random RLE string of the specified length.

    :param string_length: The desired length of the RLE string
    :param pixel_length: The desired length of the pixel array
    :param seed: The seed for the random number generator
    :return: A tuple of a randomly generated RLE string and the corresponding pixel array
    """
    if seed is not None:
        random.seed(seed)

    rle_string = ''
    pixels = []
    while len(rle_string) < string_length and len(pixels) < pixel_length:
        digit = str(random.randint(0, 9))
        run_length = random.randint(1, 27)

        if run_length > 1:
            alpha_char = chr(ord('a') + (run_length - 2))
            rle_string += alpha_char + digit
        else:
            rle_string += digit

        pixels = decode_rle_row_inner(rle_string)

    return (rle_string, pixels)

def generate_dataset_item(seed):
    string_length = 50 
    max_pixel_length = 100 
    pixel_length = random.Random(seed + 1000).randint(1, max_pixel_length)

    output_formats = [
        'pixels', 
        'json',
        'length',
        'histogram',
        'reverse',
    ]
    output_format_weights = [45, 45, 10, 30, 20]
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
        'SIMONARCRLEROW',
        'Simon-ARC-RLE-Row',
        'SimonsRLERow',
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

    instructions_length = [
        f'Length of deserialized {name_input}',
        f'length of deserialized {name_input}',
        f'Length after deserializing {name_input}',
        f'length after deserializing {name_input}',
        f'Pixel count of {name_input}',
        f'pixel count of {name_input}',
        f'Number of pixels of {name_input}',
        f'convert {name_input} and return number of pixels',
        f'Convert {name_input} and return number of pixels',
        f'Process {name_input} and return number of pixels',
        f'process {name_input} and return number of pixels',
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

    instructions_reverse = [
        f'Reverse the {name_input}',
        f'reverse the {name_input}',
        f'Flipx {name_input}',
        f'flipx {name_input}',
        f'Flip-x {name_input}',
        f'flip-x {name_input}',
    ]

    instructions = instructions_input_output
    if output_format == 'length':
        instructions = instructions_length
    if output_format == 'histogram':
        instructions = instructions_histogram
    if output_format == 'reverse':
        instructions = instructions_reverse

    instruction = random.Random(seed + 1005).choice(instructions)

    rle_string, pixels = generate_rle_string(string_length=string_length, seed=seed + 1006, pixel_length=pixel_length)

    output = None
    if output_format == 'pixels':
        output = ''.join(map(str, pixels))
    else:
        if output_format == 'json':
            output = json.dumps(list(pixels), separators=(',', ':'))
        else:
            if output_format == 'length':
                output = str(len(pixels))
            else:
                if output_format == 'histogram':
                    image = image_create(1, len(pixels), 255)
                    image[0:len(pixels), 0] = pixels
                    output = pretty_histogram_of_image(image)
                else:
                    if output_format == 'reverse':
                        pixels.reverse()
                        output = rle_serialize_line_inner(pixels)
                    else:
                        raise Exception("Unreachable code reached")

    dict = {
        'instruction': instruction,
        'input': rle_string,
        'output': output
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=100000):
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
filename = 'data_deserialize_simple.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

