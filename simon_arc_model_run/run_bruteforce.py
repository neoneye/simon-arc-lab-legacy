from datetime import datetime
import os
import sys
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from simon_arc_lab.task import Task
from simon_arc_lab.taskset import TaskSet
from simon_arc_lab.gallery_generator import gallery_generator_run
from simon_arc_lab.show_prediction_result import show_prediction_result, show_multiple_images

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Run id: {run_id}")

path_to_arc_dataset_collection_dataset = '/Users/neoneye/git/arc-dataset-collection/dataset'
if not os.path.isdir(path_to_arc_dataset_collection_dataset):
    print(f"ARC dataset collection directory '{path_to_arc_dataset_collection_dataset}' does not exist.")
    sys.exit(1)

groupname_pathtotaskdir_list = [
    ('arcagi_training', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/training')),
    # ('arcagi_evaluation', os.path.join(path_to_arc_dataset_collection_dataset, 'ARC/data/evaluation')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data')),
    # ('tama', os.path.join(path_to_arc_dataset_collection_dataset, 'arc-dataset-tama/data/symmetry_rect_input_image_and_extract_a_particular_tile')),
    # ('miniarc', os.path.join(path_to_arc_dataset_collection_dataset, 'Mini-ARC/data')),
    # ('conceptarc', os.path.join(path_to_arc_dataset_collection_dataset, 'ConceptARC/data')),
    # ('testdata', os.path.join(PROJECT_ROOT, 'testdata', 'ARC-AGI/data')),
]

for groupname, path_to_task_dir in groupname_pathtotaskdir_list:
    if not os.path.isdir(path_to_task_dir):
        print(f"path_to_task_dir directory '{path_to_task_dir}' does not exist.")
        sys.exit(1)

def datapoints_from_image(pair_id: int, image: np.array) -> list:
    height, width = image.shape
    data = []
    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]
            values = [
                pair_id,
                pixel_value,
                x,
                y,
                height,
                width,
            ]
            data.append(values)
    return data

def process_task(task: Task, weights: np.array, save_dir: str):
    # print(f"Processing task '{task.metadata_task_id}'")
    input_data = []
    for i in range(task.count_examples + task.count_tests):
        image = task.input_images[i]
        input_data += datapoints_from_image(i, image)

    target_data = []
    for i in range(task.count_examples):
        image = task.output_images[i]
        target_data += datapoints_from_image(i, image)

    random.Random(0).shuffle(input_data)
    random.Random(1).shuffle(target_data)
    # print(f"input_data: {len(input_data)} target_data: {len(target_data)}")

    # The input_data and target_data have different lengths. Sample N items from both lists, until all items have been processed.

    # Sample max N times per item.
    input_data_sample_count = np.zeros(len(input_data), dtype=int)
    target_data_sample_count = np.zeros(len(target_data), dtype=int)

    # The unvisited indexes.
    input_data_indexes = np.arange(len(input_data))
    target_data_indexes = np.arange(len(target_data))

    number_of_values_per_sample = 10
    number_of_samples = 300

    input_target_pairs = []
    for i in range(number_of_samples):
        if len(input_data_indexes) < number_of_values_per_sample:
            break

        input_data_sample_indexes = np.random.choice(input_data_indexes, number_of_values_per_sample)
        for index in input_data_sample_indexes:
            input_data_sample_count[index] += 1
            if input_data_sample_count[index] == number_of_values_per_sample:
                input_data_indexes = np.delete(input_data_indexes, np.where(input_data_indexes == index))

        # print(f"input_data_sample_indexes: {input_data_sample_indexes}")
        input_data_samples = [input_data[index] for index in input_data_sample_indexes]
        # print(f"input_data_samples: {input_data_samples}")

        if len(target_data_indexes) < number_of_values_per_sample:
            break

        target_data_sample_indexes = np.random.choice(target_data_indexes, number_of_values_per_sample)
        for index in target_data_sample_indexes:
            target_data_sample_count[index] += 1
            if target_data_sample_count[index] == number_of_values_per_sample:
                target_data_indexes = np.delete(target_data_indexes, np.where(target_data_indexes == index))
        
        # print(f"target_data_sample_indexes: {target_data_sample_indexes}")
        target_data_samples = [target_data[index] for index in target_data_sample_indexes]
        # print(f"target_data_samples: {target_data_samples}")

        if len(input_data_samples) != len(target_data_samples):
            raise ValueError(f"input and target values have different lengths. input len: {len(input_data_samples)} target len: {len(target_data_samples)}")
        
        input_target_pairs.append((input_data_samples, target_data_samples))

    xs = []
    ys = []
    count_correct = 0
    count_total = 0
    for input_data_samples, target_data_samples in input_target_pairs:
        if len(input_data_samples) != len(target_data_samples):
            raise ValueError(f"input and target values have different lengths. input len: {len(input_data_samples)} target len: {len(target_data_samples)}")
        
        n = len(input_data_samples)
        # print(f"n: {n}")
        # create a N x N matrix of the input and target values.
        matrix = np.zeros((n, n), dtype=float)
        this_count_correct = 0
        for y in range(n):
            is_target_correct = False
            for x in range(n):
                input_values = input_data_samples[y]
                target_values = target_data_samples[x]

                # print(f"input_values: {input_values} target_values: {target_values}")
                # measure correlation

                dx = input_values[2] - target_values[2]
                dy = input_values[3] - target_values[3]

                # w0 = weights[0, x]
                # w1 = weights[1, x]
                # w2 = weights[1, x]

                # a = input_values[1] * w0
                # b = dx * w1
                # c = dy * w2
                # d = a / (a + b + c + 1)

                input_pair_index = input_values[0]
                input_value = input_values[1]
                input_x = input_values[2]
                input_y = input_values[3]
                input_height = input_values[4]
                input_width = input_values[5]

                target_pair_index = target_values[0]
                target_value = target_values[1]
                target_x = target_values[2]
                target_y = target_values[3]
                target_height = target_values[4]
                target_width = target_values[5]

                is_correct = input_value == target_value

                xs_item = [
                    input_pair_index,
                    input_value,
                    input_x,
                    input_y,
                    input_height,
                    input_width,
                    target_pair_index,
                    target_value,
                    target_x,
                    target_y,
                    target_height,
                    target_width,
                    dx,
                    dy,
                ]
                ys_item = 0 if is_correct else 1

                xs.append(xs_item)
                ys.append(ys_item)

                matrix_value = 0.0
                if is_correct:
                    matrix_value = 1.0
                    is_target_correct = True

                # print(f"input_value: {input_value} target_value: {target_value}")
                matrix[y, x] = matrix_value
            if is_target_correct:
                this_count_correct += 1
        
        count_correct += (this_count_correct / n)
        count_total += 1

        # print(matrix)
    
    if count_total == 0:
        raise ValueError(f"count_total is zero")

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(xs, ys)

    # Run classifier on the test input
    # probabilities = clf.predict_proba(xs)


    average = count_correct / count_total
    # print(f"average: {average}")
    # print(f"count_correct: {count_correct} of {n}")

    # Save the image to disk or show it.
    for pair_index in range(task.count_examples):
        title = f"Task {task.metadata_task_id} pair {pair_index} average: {average:.2f}"
        input_image = task.input_images[pair_index]
        output_image = task.output_images[pair_index]
        title_image_list = [
            ('arc', 'input', input_image),
            ('arc', 'output', output_image),
        ]
        filename = f'{task.metadata_task_id}_pair{pair_index}.png'
        image_file_path = os.path.join(save_dir, filename)
        show_multiple_images(title_image_list, title=title, save_path=image_file_path)
        if pair_index >= 0:
            break

    return average


weights_width = 100
weights_height = 100
weights = np.random.rand(weights_height, weights_width)

number_of_items_in_list = len(groupname_pathtotaskdir_list)
for index, (groupname, path_to_task_dir) in enumerate(groupname_pathtotaskdir_list):
    save_dir = f'run_tasks_result/{run_id}/{groupname}'
    print(f"Processing {index+1} of {number_of_items_in_list}. Group name '{groupname}'. Results will be saved to '{save_dir}'")
    os.makedirs(save_dir, exist_ok=True)

    taskset = TaskSet.load_directory(path_to_task_dir)
    taskset.remove_tasks_by_id(set(['1_3_5_l6aejqqqc1b47pjr5g4']), True)


    # put the average in k bins
    bins = 10
    bin_width = 1 / bins
    bin_values = np.zeros(bins, dtype=float)

    for task_index, task in enumerate(taskset.tasks):
        try:
            average = process_task(task, weights, save_dir)
        except Exception as e:
            print(f"Error processing task {task.metadata_task_id}: {e}")
            continue
        bin_index = int(average / bin_width)
        bin_values[bin_index] += 1
        if task_index > 3:
            break

    print(f"bin_values: {bin_values}")

    #gallery_title = f'{groupname}, {run_id}'
    #gallery_generator_run(save_dir, title=gallery_title)
