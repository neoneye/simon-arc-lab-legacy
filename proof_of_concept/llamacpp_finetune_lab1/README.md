# Fine tune with Llama3 

### Convert to Llama3 prompt format

```
PROMPT> python convert_llama3.py
```

### Run finetune

Takes 20 minutes.

```
PROMPT> /Users/neoneye/nobackup/git/llama.cpp/finetune --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora-out lora.bin --train-data train.txt --sample-start '<SFT>' --adam-iter 1024
```
