from random_data import generate_random_byte_array
import json
import os
import random
import base64

def base64_encode_byte_array(byte_array):
    return base64.b64encode(byte_array).decode('utf-8')

def generate_dataset_item(seed):
    length = random.Random(seed + 1000).randint(0, 127)
    byte_array = generate_random_byte_array(length=length, seed=seed + 1001)

    output_formats = [
        'hex', 
        'json'
    ]
    output_format = random.Random(seed + 1002).choice(output_formats)

    names_hex = [
        'Hexadecimal',
        'hexadecimal',
        'hex',
        'Hex',
        'HEX',
    ]
    names_json = [
        'Json',
        'json',
        'JSON',
    ]

    name_output = None
    if output_format == 'hex':
        name_output = random.Random(seed + 1003).choice(names_hex)
    else:
        if output_format == 'json':
            name_output = random.Random(seed + 1004).choice(names_json)

    prefixes = [
        'Decode Base64 to ',
        'decode base64 to ',
        'Convert Base64 to ',
        'Convert BASE64 to ',
        'Transform Base64 to ',
        'Transform base64 to ',
        'Change base64 to ',
        'Change Base64 to ',
        'Change BASE64 to ',
        'Base64 to ',
        'base64 to ',
    ]

    prefix = random.Random(seed + 1005).choice(prefixes)

    instruction = prefix + name_output

    input = base64_encode_byte_array(byte_array)

    output = None
    if output_format == 'hex':
        output = byte_array.hex()
    else:
        if output_format == 'json':
            output = json.dumps(list(byte_array), separators=(',', ':'))

    dict = {
        'instruction': instruction,
        'input': input,
        'output': output
    }
    return dict

def generate_dataset(max_num_samples=1000, max_byte_size=1024*1024, seed_start=0):
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
    max_num_samples=20000,
    max_byte_size=1024*1024*10,
)

# Save dataset to file
filename = 'base64decode_dataset.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

