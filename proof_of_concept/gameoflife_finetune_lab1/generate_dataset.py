from game_of_life import GameOfLife
from random_game_of_life import generate_random_game_of_life_string
from game_of_life_mutator import GameOfLifeMutator
import json
import os
import random

def shuffle_instruction(seed, dead, alive, wrap, iterations):
    params = [f"dead='{dead}'", f"alive='{alive}'", f"wrap={wrap}"]
    if iterations > 1:
        params.append(f"iterations={iterations}")
    random.Random(seed).shuffle(params)
    return f"Game of Life. " + ' '.join(params)

def generate_dataset_item(seed):
    wrap_x = False
    if seed & 1 == 0:
        wrap_x = True
    wrap_y = False
    if seed & 2 == 0:
        wrap_y = True
    iterations = ((seed >> 2) & 1) + 1

    junk_spaces_in_input = seed % 13

    input = generate_random_game_of_life_string(seed=seed, min_width=5, max_width=15, min_height=5, max_height=15)
    gol = GameOfLife.create(input, wrap_x=wrap_x, wrap_y=wrap_y, iterations=iterations)
    output = gol.output_str
    mutator = GameOfLifeMutator()
    mutated_input = mutator.mutate(input, num_extra_spaces=junk_spaces_in_input, seed=seed)
    mutated_output = mutator.mutate(output, seed=seed)
    input_state = mutated_input['mutated_str'] 
    output_state = mutated_output['mutated_str']
    dead = mutated_input['zero_replacement']
    alive = mutated_input['one_replacement']

    wrap = 'none'
    if wrap_x and wrap_y:
        wrap = 'xy'
    elif wrap_x:
        wrap = 'x'
    elif wrap_y:
        wrap = 'y'

    instruction_seed = seed + 1000  # Ensure a different seed for shuffling
    instruction = shuffle_instruction(instruction_seed, dead, alive, wrap, iterations)
    dict = {
        'instruction': instruction,
        'input': input_state,
        'output': output_state
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
    max_num_samples=50000,
    max_byte_size=1024*1024*10,
)

# Save dataset to file
filename = 'game_of_life_dataset.jsonl'
with open(filename, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

# Summary
file_size = os.path.getsize(filename)
print(f"Generated {len(dataset)} samples, saved to {filename}, file size: {file_size} bytes.")

