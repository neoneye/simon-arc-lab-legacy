from game_of_life import game_of_life
from random_game_of_life import generate_random_game_of_life_string
from game_of_life_mutator import GameOfLifeMutator
import json
import os

def generate_dataset_item(seed):
    wrap_x = False
    if seed & 1 == 0:
        wrap_x = True
    wrap_y = False
    if seed & 2 == 0:
        wrap_y = True
    input = generate_random_game_of_life_string(seed=seed, min_width=5, max_width=15, min_height=5, max_height=15)
    output = game_of_life(input, wrap_x=wrap_x, wrap_y=wrap_y)
    mutator = GameOfLifeMutator()
    mutated_input = mutator.mutate(input, seed=seed)
    mutated_output = mutator.mutate(output, seed=seed)
    input_state = mutated_input['mutated_str'] 
    output_state = mutated_output['mutated_str']
    dead = mutated_input['zero_replacement']
    alive = mutated_input['one_replacement']
    instruction = f"Game of Life."
    instruction += f" dead='{dead}'"
    instruction += f" alive='{alive}'"
    if wrap_x and wrap_y:
        instruction += " wrap=xy"
    elif wrap_x:
        instruction += " wrap=x"
    elif wrap_y:
        instruction += " wrap=y"
    else:
        instruction += " wrap=none"
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

