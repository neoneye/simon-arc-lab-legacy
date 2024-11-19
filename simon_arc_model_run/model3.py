from datetime import datetime
import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from transformer_asarkar import Transformer

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    # ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data/symmetry_rect_input_image_and_extract_a_particular_tile')),
    ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

def process_task(task: Task):
    print(f"Processing task '{task.metadata_task_id}'")
    input_data = []
    for i in range(task.count_examples + task.count_tests):
        image = task.input_images[i]
        height, width = image.shape
        for y in range(height):
            for x in range(width):
                pixel_value = image[y, x]
                values = [
                    i,
                    pixel_value,
                    x,
                    y,
                    height,
                    width,
                ]
                input_data.append(values)

    target_data = []
    for i in range(task.count_examples):
        image = task.output_images[i]
        height, width = image.shape
        for y in range(height):
            for x in range(width):
                pixel_value = image[y, x]
                values = [
                    i,
                    pixel_value,
                    x,
                    y,
                    height,
                    width,
                ]
                target_data.append(values)

    random.Random(0).shuffle(input_data)
    random.Random(1).shuffle(target_data)
    # print(f"input_data: {len(input_data)} target_data: {len(target_data)}")


def prepare_data():
    number_of_items_in_list = len(groupname_pathtotaskdir_list)
    for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
        save_dir = f'run_tasks_result/{run_id}/{groupname}'
        print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

        taskset = TaskSet.load_directory(path_to_task_dir)
        taskset.remove_tasks_by_id(set(['1_3_5_l6aejqqqc1b47pjr5g4']), True)


        for task in taskset.tasks:
            process_task(task)

#prepare_data()
#exit(1)

src_vocab_size = 15 # 10 colors + special tokens
tgt_vocab_size = 15 # 10 colors + special tokens
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 25 # 5x5 pixels
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

if False:
    # Calculate total and trainable parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # Parameter breakdown by component
    print("\nParameter breakdown by component:")
    for name, param in transformer.named_parameters():
        print(f"{name}: {param.numel()} parameters")

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

