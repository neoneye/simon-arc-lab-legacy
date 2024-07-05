import json
import os
import random
from deserialize import decode_rle_row_inner
from serialize import serialize, rle_serialize_line_inner

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

    rle_string2 = rle_serialize_line_inner(pixels)

    return (rle_string2, pixels)

def generate_dataset_item(seed):
    string_length = 50 
    max_pixel_length = 100 
    pixel_length = random.Random(seed + 1000).randint(1, max_pixel_length)

    input_formats = [
        'pixels', 
        'json'
    ]
    input_format_weights = [0.5, 0.5]
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
        'SIMONARCRLEROW',
        'Simon-ARC-RLE-Row',
        'SimonsRLERow',
    ]
    name_output = random.Random(seed + 1004).choice(name_outputs)

    instructions = [
        f'Serialize {name_input} to {name_output}',
        f'Serialize {name_input} to {name_output}',
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

    rle_string, pixels = generate_rle_string(string_length=string_length, seed=seed + 1006, pixel_length=pixel_length)

    input = None
    if input_format == 'pixels':
        input = ''.join(map(str, pixels))
    else:
        if input_format == 'json':
            input = json.dumps(list(pixels), separators=(',', ':'))
        else:
            input = str(len(pixels))

    dict = {
        'instruction': instruction,
        'input': input,
        'output': rle_string
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=200000):
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
filename = 'data_serialize_simple.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

