import re
import my_arc_thing
import my_arc_thing.arc_json_model as ajm
import os
import json
import datetime
from llama_cpp import Llama
from tqdm import tqdm

def format_image_as_compact_json(image):
    return json.dumps(image.pixels.tolist(), separators=(',', ':'))

def format_image_as_compact_json_with_greek_alphabet(image):
    s = format_image_as_compact_json(image)
    # regex replace digit followed by comma, with digit only
    s = re.sub(r"(\d),", r"\1", s)
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("0", "α") # alpha
    s = s.replace("1", "β") # beta
    s = s.replace("2", "γ") # gamma
    s = s.replace("3", "δ") # delta
    s = s.replace("4", "ε") # epsilon
    s = s.replace("5", "ζ") # zeta
    s = s.replace("6", "η") # eta
    s = s.replace("7", "θ") # theta
    s = s.replace("8", "ι") # iota
    s = s.replace("9", "κ") # kappa
    return s

def format_image_as_compact_json_with_cycled_digits(image):
    s = format_image_as_compact_json(image)
    # regex replace each digit with (digit+1)%10
    s = re.sub(r"\d", lambda x: str((int(x.group())+1)%10), s)
    return s

def format_task_as_prompt(task):
    prompt = "Solve this ARC task\n"
    expected_response_text = ""
    count_test = 0
    for pair_index, pair in enumerate(task.pairs):
        input_json = format_image_as_compact_json(pair.input)
        output_json = format_image_as_compact_json(pair.output)
        # input_json = format_image_as_compact_json_with_greek_alphabet(pair.input)
        # output_json = format_image_as_compact_json_with_greek_alphabet(pair.output)
        # input_json = format_image_as_compact_json_with_cycled_digits(pair.input)
        # output_json = format_image_as_compact_json_with_cycled_digits(pair.output)
        if pair.pair_type == ajm.PairType.TRAIN:
            prompt += f"input {pair_index}\n{input_json}\noutput {pair_index}\n{output_json}\n"
        if pair.pair_type == ajm.PairType.TEST:
            count_test += 1
            if count_test == 1:
                prompt += f"input {pair_index}\n{input_json}\noutput {pair_index}\n"
                expected_response_text = output_json
            else:
                print(f"Skipping task with 2 or more test pairs. task: {task}")
    return (prompt, expected_response_text)

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
    #pbar.write(f"Processing: {file_path}")
    task = ajm.Task.load(file_path)
    prompt, expected_response_text = format_task_as_prompt(task)
    if len(prompt) > 512:
        return

    response_dict = llm(prompt, max_tokens=1024, stop=["\ninput"], temperature=0.0)
    #pbar.write(f"response dict: {response}")

    s = f"# ARC Task {file_index}\n\n"
    s += f"original path: {file_path}\n\n"
    s += f"prompt:\n{prompt}\n\n"
   
    s += f"expected response text:\n{expected_response_text}\n\n"

    actual_response_text = response_dict["choices"][0]["text"]
    s += f"actual response text:\n{actual_response_text}\n\n"

    is_correct = expected_response_text == actual_response_text
    if is_correct:
        s += f"status: correct\n\n"
    else:
        s += f"status: incorrect\n\n"

    s += f"response dict:\n{response_dict}\n\n"

    response_filename = f"{file_index}.md"
    response_path = os.path.join(output_dir, response_filename)
    #print(f"Writing response to: {response_path}")
    with open(response_path, 'w') as f:
       f.write(s)
    
    #pbar.write(f"index: {file_index}  bytes: {len(prompt)}")
    #if is_correct:
    #    pbar.write(f"response matches expected!")

def summarize_results(output_dir):
    count_correct = 0
    for filename in os.listdir(output_dir):
        if filename.endswith(".md"):
            with open(os.path.join(output_dir, filename), 'r') as file:
                if "status: correct" in file.read():
                    count_correct += 1
    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"Number of 'correct' responses: {count_correct}\n")

def main():
    root_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data'
    output_dir = create_dir_for_today()
    json_file_paths = get_json_file_paths(root_dir)

    if not json_file_paths:
        print("No JSON files found.")
        return

    model_path = "/Users/neoneye/nobackup/git/llama.cpp/models/llama-2-7b/llama-2-7b.Q4_0.gguf"
    llm = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False)

    with tqdm(json_file_paths, desc="Processing JSON files") as pbar:
        for index, file_path in enumerate(pbar):
            process_json_file(llm, file_path, index, pbar, output_dir)
    
    summarize_results(output_dir)

if __name__ == "__main__":
    main()
