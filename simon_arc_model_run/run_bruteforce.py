from datetime import datetime
import os
import sys
import numpy as np
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run

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
    random.Random(0).shuffle(target_data)
    print(f"input_data: {len(input_data)} target_data: {len(target_data)}")

    # The input_data and target_data have different lengths. Sample N items from both lists, until all items have been processed.

    # Sample max N times per item.
    input_data_sample_count = np.zeros(len(input_data), dtype=int)
    target_data_sample_count = np.zeros(len(target_data), dtype=int)

    # The unvisited indexes.
    input_data_indexes = np.arange(len(input_data))
    target_data_indexes = np.arange(len(target_data))

    number_of_values_per_sample = 10
    number_of_samples = 3

    for i in range(number_of_samples):
        input_data_sample_indexes = np.random.choice(input_data_indexes, number_of_values_per_sample)
        for index in input_data_sample_indexes:
            input_data_sample_count[index] += 1
            if input_data_sample_count[index] == number_of_values_per_sample:
                input_data_indexes = np.delete(input_data_indexes, np.where(input_data_indexes == index))

        # print(f"input_data_sample_indexes: {input_data_sample_indexes}")
        input_values = [input_data[index] for index in input_data_sample_indexes]
        # print(f"input_values: {input_values}")

        target_data_sample_indexes = np.random.choice(target_data_indexes, number_of_values_per_sample)
        for index in target_data_sample_indexes:
            target_data_sample_count[index] += 1
            if target_data_sample_count[index] == number_of_values_per_sample:
                target_data_indexes = np.delete(target_data_indexes, np.where(target_data_indexes == index))
        
        # print(f"target_data_sample_indexes: {target_data_sample_indexes}")
        target_values = [target_data[index] for index in target_data_sample_indexes]
        # print(f"target_values: {target_values}")

        print(f"input len: {len(input_values)} target len: {len(target_values)}")

    exit(1)


weights_width = 100
weights_height = 100
weights = np.random.rand(weights_height, weights_width)

number_of_items_in_list = len(groupname_pathtotaskdir_list)
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")

    taskset = TaskSet.load_directory(path_to_task_dir)

    for task in taskset.tasks:
        process_task(task)

    #gallery_title = f'{groupname}, {run_id}'
    #gallery_generator_run(save_dir, title=gallery_title)
