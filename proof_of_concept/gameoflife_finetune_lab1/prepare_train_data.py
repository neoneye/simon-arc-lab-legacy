import json

def convert_to_format_v1(input_file, output_file):
    system = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.'
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            instruction = data['instruction']
            input_state = data['input']
            output_state = data['output']
            user = f"### Instruction:\n{instruction}\n### Input:\n{input_state}"
            assistant = output_state
            prompt = f"<SFT><|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|><|start_header_id|>user<|end_header_id|>{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{assistant}"
            f_out.write(prompt)

def convert_to_format_v2(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            instruction = data['instruction']
            input_state = data['input']
            output_state = data['output']
            system = instruction
            user = input_state
            assistant = output_state
            prompt = f"<SFT><|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|><|start_header_id|>user<|end_header_id|>{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{assistant}"
            f_out.write(prompt)

def convert_to_format_v3(input_file, output_file):
    system = 'You are a helpful assistant.'
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            instruction = data['instruction']
            input_state = data['input']
            output_state = data['output']
            user = f"{instruction}\n\n{input_state}"
            assistant = output_state
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|><|start_header_id|>user<|end_header_id|>{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{assistant}<|eot_id|>"
            f_out.write(prompt)

def convert_to_format_v4(input_file, output_file):
    system = 'You are a helpful assistant.'
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            instruction = data['instruction']
            input_state = data['input']
            output_state = data['output']
            user = f"{instruction}\n\n# Input\n{input_state}\nOutput"
            assistant = output_state
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|><|start_header_id|>user<|end_header_id|>{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{assistant}<|eot_id|>"
            f_out.write(prompt)

def convert_to_format_v5(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            instruction = data['instruction']
            input_state = data['input']
            output_state = data['output']
            prompt = f"<s>{instruction}\n\n# Input\n{input_state}\n\n# Output\n{output_state}\n\n"
            f_out.write(prompt)

def convert_to_format_v6(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            instruction = data['instruction']
            input_state = data['input']
            output_state = data['output']
            text = f"{instruction}\n\n# Input\n{input_state}\n\n# Output\n{output_state}\n\n"
            item = { 'text': text }
            f_out.write(json.dumps(item) + '\n')

input_file = 'game_of_life_dataset.jsonl'
output_file = 'train_data.jsonl'
convert_to_format_v6(input_file, output_file)

print(f"Converted dataset saved to {output_file}")
