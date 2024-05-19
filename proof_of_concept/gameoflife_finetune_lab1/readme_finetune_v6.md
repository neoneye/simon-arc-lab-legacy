# Finetune v5

v6: I'm trying out Apple's MLX "lora" for finetuning.
v5+v4+v3+v2+v1: I used the llama.cpp's "finetune" command.

```
(venv) PROMPT> ./prepare_train_data.py
```

From my game of life project, from the `train_data.jsonl`, I have manually extracted 
1000 lines into a train.jsonl file.
100 lines into a test.jsonl file.
100 lines into a valid.jsonl file.

Place these files inside: `/Users/neoneye/Downloads/mlx-examples-main/lora/data`

Change dir to the MLX example repo: `/Users/neoneye/Downloads/mlx-examples-main/lora`

```
(venv) PROMPT> python lora.py --model mlx_model --train --iters 600
```
