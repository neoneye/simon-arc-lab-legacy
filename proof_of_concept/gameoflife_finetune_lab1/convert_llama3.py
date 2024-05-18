import json

def convert_to_llama_format_v1(input_file, output_file):
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

def convert_to_llama_format_v2(input_file, output_file):
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

input_file = 'game_of_life_dataset.jsonl'
output_file = 'game_of_life_llama3_prompts.txt'
convert_to_llama_format_v2(input_file, output_file)

print(f"Converted dataset saved to {output_file}")
