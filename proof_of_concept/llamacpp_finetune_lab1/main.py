import my_arc_thing
import my_arc_thing.arc_json_model as ajm
import os
import json
from llama_cpp import Llama
from tqdm import tqdm  # Import tqdm

def format_image_as_compact_json(image):
    """
    Format an Image object as a JSON-serializable dictionary, compacted without spaces.
    """
    # Convert image pixels to list and use json.dumps to serialize without spaces
    return json.dumps(image.pixels.tolist(), separators=(',', ':'))

def format_task_as_prompt(task):
    prompt = "# Solve this ARC task\n"
    for pair_index, pair in enumerate(task.pairs):
        input_json = format_image_as_compact_json(pair.input)
        output_json = format_image_as_compact_json(pair.output)
        if pair.pair_type == ajm.PairType.TRAIN:
            prompt += f"Input[{pair_index}]:\n{input_json}\nOutput[{pair_index}]:\n{output_json}\n"
        if pair.pair_type == ajm.PairType.TEST:
            prompt += f"Input[{pair_index}]:\n{input_json}\nOutput[{pair_index}]:\nPREDICT\n"
    return prompt

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

def process_json_file(llm, file_path, pbar):
    """
    Load the ARC task, format it into a prompt, query the LLM, and save the markdown response.
    """
    pbar.write(f"Processing: {file_path}")  # Use pbar.write instead of print
    task = ajm.Task.load(file_path)
    
    prompt = format_task_as_prompt(task)
    print(prompt)
    #return

    # if length of prompt is too long, 512 bytes, then skip
    if len(prompt) > 512:
        return

    # Query the LLM
    response = llm(
        prompt,
        max_tokens=1024,  # You might adjust this depending on your needs
        stop=["\n\n"],  # Stop generation on two newlines, or other suitable delimiter
        echo=True
    )
    
    # Use pbar.write to display the response without interrupting the progress bar
    pbar.write(f"response dict: {response}")
    #response_filename = os.path.splitext(os.path.basename(file_path))[0] + ".md"
    #response_path = os.path.join(os.path.dirname(file_path), response_filename)
    #with open(response_path, 'w') as f:
    #    f.write(response)

def main():
    root_dir = '/Users/neoneye/git/arc-dataset-collection/dataset/ARC/data'
    json_file_paths = get_json_file_paths(root_dir)
    
    # Optionally, perform sanity checks on the collected paths
    if not json_file_paths:
        print("No JSON files found.")
        return

    model_path = "/Users/neoneye/nobackup/git/llama.cpp/models/llama-2-7b/llama-2-7b.Q4_0.gguf"
    
    # Initialize the LLaMA model
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,  # Uncomment to use GPU acceleration
        # Additional configuration can be set here if needed
    )
    
    # Process each JSON file with tqdm progress bar
    with tqdm(json_file_paths, desc="Processing JSON files") as pbar:
        for file_path in pbar:
            process_json_file(llm, file_path, pbar)

if __name__ == "__main__":
    main()
