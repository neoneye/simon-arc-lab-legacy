import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(torch.backends.mps.is_available())

#torch.set_default_device("mps")


base_model_name = "/Users/neoneye/nobackup/git/llama.cpp/models/llama-3-8b-bnb-4bit"

# bnb_config = BitsAndBytesConfig(
#     #load_in_4bit=True,
#     load_in_4bit=False,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

#base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map='mps')
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='mps')
#tokenizer = AutoTokenizer.from_pretrained(base_model_name)

"""
Unused kwargs: ['_load_in_4bit', '_load_in_8bit', 'quant_method']. These kwargs are not used in <class 'transformers.utils.quantization_config.BitsAndBytesConfig'>.
Traceback (most recent call last):
  File "/Users/neoneye/git/python_arc/proof_of_concept/merge_lora_with_base/main.py", line 4, in <module>
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
  File "/Users/neoneye/git/python_arc/proof_of_concept/merge_lora_with_base/venv/lib/python3.9/site-packages/transformers/models/auto/auto_factory.py", line 563, in from_pretrained
    return model_class.from_pretrained(
  File "/Users/neoneye/git/python_arc/proof_of_concept/merge_lora_with_base/venv/lib/python3.9/site-packages/transformers/modeling_utils.py", line 3202, in from_pretrained
    hf_quantizer.validate_environment(
  File "/Users/neoneye/git/python_arc/proof_of_concept/merge_lora_with_base/venv/lib/python3.9/site-packages/transformers/quantizers/quantizer_bnb_4bit.py", line 62, in validate_environment
    raise RuntimeError("No GPU found. A GPU is needed for quantization.")
RuntimeError: No GPU found. A GPU is needed for quantization.
"""

