import my_arc_thing
import my_arc_thing.arc_json_model as ajm
import os
import json

def process_json_file(file_path):
    """
    Load the ARC task.
    Check if it gets solved correctly by the LLM.
    """
    print(f"Processing: {file_path}")
    task = ajm.Task.load(file_path)
    print(task)

def main():
    root_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data'

    for subdir, dirs, files in os.walk(root_dir):
        files = sorted(files)
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                process_json_file(file_path)

if __name__ == "__main__":
    main()
