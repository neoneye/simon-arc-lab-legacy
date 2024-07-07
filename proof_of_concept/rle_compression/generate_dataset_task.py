# IDEA: extract output A
# IDEA: histogram of input B
# IDEA: rotate cw output D
# IDEA: flipx input E
# IDEA: compare input E and output E
# IDEA: intersection of all input histograms
# IDEA: union of all input histograms
# IDEA: intersection of all output histograms
# IDEA: are the sizes of the input and output the same?
# IDEA: union of all output histograms
import json
import os
import random
import numpy as np
from deserialize import deserialize
from serialize import serialize
from image_util import *
from image_create_random_advanced import image_create_random_advanced

class MyTask:
    def __init__(self):
        self.input_images = []
        self.output_images = []
        self.count_examples = 0
        self.count_tests = 0

    def append_pair(self, input_image, output_image, is_example):
        self.assert_count()
        self.input_images.append(input_image)
        self.output_images.append(output_image)
        if is_example:
            self.count_examples += 1
        else:
            self.count_tests += 1
        self.assert_count()

    def assert_count(self):
        assert len(self.input_images) == len(self.output_images)
        assert self.count_examples + self.count_tests == len(self.input_images)

    def input_ids(self):
        self.assert_count()
        names = []
        for i in range(len(self.input_images)):
            if i < self.count_examples:
                name = "Example"
            else:
                name = "Test"
            names.append(f"Input {i} {name}")
        return names

    def output_ids(self):
        self.assert_count()
        names = []
        for i in range(len(self.input_images)):
            if i < self.count_examples:
                name = "Example"
            else:
                name = "Test"
            names.append(f"Output {i} {name}")
        return names

    def to_string(self):
        self.assert_count()
        input_ids = self.input_ids()
        output_ids = self.output_ids()
        s = ""
        for i in range(len(self.input_images)):
            if i > 0:
                s += "\n"
            s += input_ids[i] + "\n"
            s += serialize(self.input_images[i]) + "\n"
            s += output_ids[i] + "\n"
            s += serialize(self.output_images[i])
        return s

def generate_task(seed):
    count_example = random.Random(seed + 1).randint(2, 5)
    count_test = random.Random(seed + 2).randint(1, 3)
    task = MyTask()

    for i in range(count_example+count_test):
        is_example = i < count_example
        input_image = image_create_random_advanced(seed + 1000 + i, 5, 10, 5, 10)
        output_image = image_create_random_advanced(seed + 2000 + i, 5, 10, 5, 10)
        task.append_pair(input_image, output_image, is_example)

    return task

def generate_dataset_item(seed):
    dataformat_names = [
        'SIMONARCTASK',
        'Simon-ARC-Task',
        'SimonsArcTask',
    ]
    dataformat_name = random.Random(seed + 1004).choice(dataformat_names)

    task = generate_task(seed)
    print("---")
    print(task.to_string())

    image_id = "Input-A"

    instructions_extract = [
        f'This is {dataformat_name} data. Extract image {image_id}',
    ]

    instructions = instructions_extract

    instruction = random.Random(seed + 1005).choice(instructions)

    input = task.to_string()

    output = ""

    dict = {
        'instruction': instruction,
        'input': input,
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
filename = 'data_task.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

