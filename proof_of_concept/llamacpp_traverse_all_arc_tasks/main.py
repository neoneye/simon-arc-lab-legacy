import re
from collections import Counter
import my_arc_thing
import my_arc_thing.arc_json_model as ajm
import my_arc_thing.image
import my_arc_thing.compress_huffman as compress_huffman
import os
import json
import datetime
import base64
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

def format_image_as_rle1(arc_image):
    image = arc_image.to_image()
    data = image.pixels_1d()
    encoding = []
    prev_char = None
    count = 1

    for char in data:
        if char != prev_char:
            if prev_char:
                encoding.append(prev_char)
                encoding.append(count)
            count = 1
            prev_char = int(char)
        else:
            count += 1

    encoding.append(prev_char)
    encoding.append(count)
    return json.dumps(encoding, separators=(',', ':'))

def format_image_as_huffman_encoded(arc_image, huffman_code_map):
    image = arc_image.to_image()
    data = image.pixels_1d()
    binary_string_data = compress_huffman.huffman_encode(data, huffman_code_map)

    # Convert the Huffman encoded binary string to bytes
    byte_array = compress_huffman.binary_string_to_bytes(binary_string_data)
    #print(f"Byte array: {byte_array}")

    # Base64 encode the byte array
    base64_encoded_data = base64.b64encode(byte_array).decode('utf-8')
    return base64_encoded_data

def format_task_as_prompt(task):
    prompt = "ARC puzzle\n"
    expected_response_text = ""
    count_test = 0

    huffman_code_map = None
    use_huffman = False
    if use_huffman:
        freq_map = Counter()
        for pair_index, pair in enumerate(task.pairs):
            if pair.pair_type == ajm.PairType.TRAIN:
                freq_map += pair.input.to_image().histogram()
                freq_map += pair.output.to_image().histogram()
            if pair.pair_type == ajm.PairType.TEST:
                freq_map += pair.input.to_image().histogram()
                # no output image for test pair

        root = compress_huffman.build_huffman_tree(freq_map)
        node_count = compress_huffman.count_huffman_nodes(root)
        prompt += f"Number of HuffmanNode objects in the tree: {node_count}\n"
        huffman_code_map = compress_huffman.create_codes(root)
        # print(f"Code Map: {huffman_code_map}")

    for pair_index, pair in enumerate(task.pairs):
        input_json = format_image_as_compact_json(pair.input)
        output_json = format_image_as_compact_json(pair.output)
        # input_json = format_image_as_compact_json_with_greek_alphabet(pair.input)
        # output_json = format_image_as_compact_json_with_greek_alphabet(pair.output)
        # input_json = format_image_as_compact_json_with_cycled_digits(pair.input)
        # output_json = format_image_as_compact_json_with_cycled_digits(pair.output)
        # input_json = format_image_as_rle1(pair.input)
        # output_json = format_image_as_rle1(pair.output)
        if use_huffman:
            input_json = format_image_as_huffman_encoded(pair.input, huffman_code_map)
            output_json = format_image_as_huffman_encoded(pair.output, huffman_code_map)
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

def process_json_file(llm, file_path, file_index, pbar, output_dir, n_ctx):
    #pbar.write(f"Processing: {file_path}")
    task = ajm.Task.load(file_path)
    prompt, expected_response_text = format_task_as_prompt(task)
    if len(prompt) > n_ctx:
        return

    response_dict = llm(prompt, max_tokens=1024, stop=["\n"], temperature=0.0)
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

    model_path = "/Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf"
    # model_path = "/Users/neoneye/.cache/lm-studio/models/lmstudio-community/llama-2-7b/llama-2-7b.Q4_0.gguf"
    #model_path = "/Users/neoneye/.cache/lm-studio/models/Qwen/Qwen1.5-7B-Chat-GGUF/qwen1_5-7b-chat-q5_k_m.gguf"
    n_ctx = 512 # takes 5 minutes
    #n_ctx = 1024 # takes 20 minutes
    #n_ctx = 2048 # takes 70 minutes
    seed = 0
    llm = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False, n_ctx=n_ctx, seed=seed)

    with tqdm(json_file_paths, desc="Processing JSON files") as pbar:
        for index, file_path in enumerate(pbar):
            process_json_file(llm, file_path, index, pbar, output_dir, n_ctx)
    
    summarize_results(output_dir)

if __name__ == "__main__":
    main()
