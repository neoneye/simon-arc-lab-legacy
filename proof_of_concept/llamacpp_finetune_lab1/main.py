import my_arc_thing
import my_arc_thing.arc_json_model as ajm
import os
import json
import datetime
from llama_cpp import Llama
from tqdm import tqdm

def format_image_as_compact_json(image):
    return json.dumps(image.pixels.tolist(), separators=(',', ':'))

def format_task_as_prompt(task):
    prompt = "Solve this ARC task\n"
    count_test = 0
    for pair_index, pair in enumerate(task.pairs):
        input_json = format_image_as_compact_json(pair.input)
        output_json = format_image_as_compact_json(pair.output)
        if pair.pair_type == ajm.PairType.TRAIN:
            prompt += f"input {pair_index}\n{input_json}\noutput {pair_index}\n{output_json}\n"
        if pair.pair_type == ajm.PairType.TEST:
            count_test += 1
            if count_test == 1:
                prompt += f"input {pair_index}\n{input_json}\noutput {pair_index}\n"
            else:
                print(f"Skipping task with 2 or more test pairs. task: {task}")
    return prompt

def create_dir_for_today():
    today_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    dir_path = f"checkpoint/{today_str}"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_json_file_paths(root_dir):
    """
    Traverse the directory and collect all JSON file paths.
    """
    json_file_paths = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in sorted(files):
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                json_file_paths.append(file_path)
    return json_file_paths

def process_json_file(llm, file_path, file_index, pbar, output_dir):
    pbar.write(f"Processing: {file_path}")
    task = ajm.Task.load(file_path)
    prompt = format_task_as_prompt(task)
    if len(prompt) > 512:
        return

    response_dict = llm(prompt, max_tokens=1024, stop=["\ninput"], temperature=0.0)
    #pbar.write(f"response dict: {response}")

    s = f"# ARC Task {file_index}\n\n"
    s += f"original path: {file_path}\n\n"
    s += f"prompt:\n{prompt}\n\n"
    s += f"response:\n{response_dict}\n\n"

    response_text = response_dict["choices"][0]["text"]
    s += f"response text:\n{response_text}\n\n"

    response_filename = f"{file_index}.md"
    response_path = os.path.join(output_dir, response_filename)
    #print(f"Writing response to: {response_path}")
    with open(response_path, 'w') as f:
       f.write(s)
    
    pbar.write(f"index: {file_index}  bytes: {len(prompt)}")

def main():
    root_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data'
    output_dir = create_dir_for_today()
    json_file_paths = get_json_file_paths(root_dir)

    if not json_file_paths:
        print("No JSON files found.")
        return

    model_path = "/Users/neoneye/nobackup/git/llama.cpp/models/llama-2-7b/llama-2-7b.Q4_0.gguf"
    llm = Llama(model_path=model_path, n_gpu_layers=-1)

    with tqdm(json_file_paths, desc="Processing JSON files") as pbar:
        for index, file_path in enumerate(pbar):
            process_json_file(llm, file_path, index, pbar, output_dir)

if __name__ == "__main__":
    main()
