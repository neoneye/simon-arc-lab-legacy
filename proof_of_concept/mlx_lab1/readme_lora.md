# Download mlx-examples repo

```
PROMPT> pwd
~/Downloads/mlx-examples-main/lora
```

# venv

```
PROMPT> python3 -m venv venv
PROMPT> source venv/bin/activate
(venv) PROMPT> pip install -r requirements.txt
```

# Quantize

```
(venv) PROMPT> python convert.py --hf-path mistralai/Mistral-7B-Instruct-v0.2 -q
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
[INFO] Loading
Fetching 11 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 153280.21it/s]
/Users/neoneye/Downloads/mlx-examples-main/lora/venv/lib/python3.12/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
[INFO] Quantizing
```

This creates a `mlx_model` dir.

# Training data formatting

The dir `data` is where the training data is located.

```json
{"text": "instruction input output"}
```

The files must be named `test.jsonl`, `train.jsonl`, `valid.jsonl`.

How are these files being used?

# Finetune

Takes jsonl files from the dir `data`.

It save checkpoints, every 100 iteration, to a file named `adaptors.npz`.

It uses max GPU, but barely any activity on the CPU cores.

It seems to have a hard limit of 2048 tokens.

```
(venv) PROMPT> python lora.py --model mlx_model --train --iters 600
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Loading pretrained model
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
Total parameters 1133.449M
Trainable parameters 1.704M
Loading datasets
Training
Iter 1: Val loss 2.681, Val took 24.821s
Iter 10: Train loss 2.254, It/sec 0.477, Tokens/sec 191.716
Iter 20: Train loss 1.439, It/sec 0.567, Tokens/sec 229.502
Iter 30: Train loss 1.368, It/sec 0.558, Tokens/sec 229.759
Iter 40: Train loss 1.279, It/sec 0.610, Tokens/sec 236.470
Iter 50: Train loss 1.204, It/sec 0.560, Tokens/sec 222.776
Iter 60: Train loss 1.076, It/sec 0.603, Tokens/sec 238.749
Iter 70: Train loss 1.141, It/sec 0.521, Tokens/sec 211.226
Iter 80: Train loss 1.154, It/sec 0.542, Tokens/sec 217.645
Iter 90: Train loss 1.117, It/sec 0.619, Tokens/sec 234.763
Iter 100: Train loss 1.165, It/sec 0.529, Tokens/sec 201.580
Iter 100: Saved adapter weights to adapters.npz.
Iter 110: Train loss 1.041, It/sec 0.612, Tokens/sec 236.853
Iter 120: Train loss 1.018, It/sec 0.560, Tokens/sec 218.791
Iter 130: Train loss 1.037, It/sec 0.540, Tokens/sec 213.813
Iter 140: Train loss 1.034, It/sec 0.597, Tokens/sec 232.795
Iter 150: Train loss 1.003, It/sec 0.525, Tokens/sec 213.561
Iter 160: Train loss 0.942, It/sec 0.563, Tokens/sec 220.990
Iter 170: Train loss 0.974, It/sec 0.557, Tokens/sec 229.496
Iter 180: Train loss 0.930, It/sec 0.645, Tokens/sec 234.304
Iter 190: Train loss 0.976, It/sec 0.567, Tokens/sec 227.233
Iter 200: Train loss 1.053, It/sec 0.585, Tokens/sec 222.719
Iter 200: Val loss 1.103, Val took 25.939s
Iter 200: Saved adapter weights to adapters.npz.
Iter 210: Train loss 1.004, It/sec 0.572, Tokens/sec 217.705
Iter 220: Train loss 0.946, It/sec 0.572, Tokens/sec 227.931
Iter 230: Train loss 0.935, It/sec 0.580, Tokens/sec 225.627
Iter 240: Train loss 0.901, It/sec 0.599, Tokens/sec 229.604
Iter 250: Train loss 0.926, It/sec 0.525, Tokens/sec 217.714
Iter 260: Train loss 0.815, It/sec 0.547, Tokens/sec 221.334
Iter 270: Train loss 0.793, It/sec 0.539, Tokens/sec 210.814
Iter 280: Train loss 0.796, It/sec 0.578, Tokens/sec 222.382
Iter 290: Train loss 0.873, It/sec 0.556, Tokens/sec 213.014
Iter 300: Train loss 0.754, It/sec 0.546, Tokens/sec 222.538
Iter 300: Saved adapter weights to adapters.npz.
Iter 310: Train loss 0.873, It/sec 0.602, Tokens/sec 225.468
Iter 320: Train loss 0.736, It/sec 0.522, Tokens/sec 203.383
Iter 330: Train loss 0.787, It/sec 0.514, Tokens/sec 207.245
Iter 340: Train loss 0.762, It/sec 0.523, Tokens/sec 201.394
Iter 350: Train loss 0.777, It/sec 0.551, Tokens/sec 211.394
Iter 360: Train loss 0.786, It/sec 0.532, Tokens/sec 212.694
Iter 370: Train loss 0.726, It/sec 0.489, Tokens/sec 200.141
Iter 380: Train loss 0.747, It/sec 0.515, Tokens/sec 197.871
Iter 390: Train loss 0.762, It/sec 0.546, Tokens/sec 211.595
Iter 400: Train loss 0.717, It/sec 0.458, Tokens/sec 194.130
Iter 400: Val loss 0.999, Val took 27.578s
Iter 400: Saved adapter weights to adapters.npz.
Iter 410: Train loss 0.683, It/sec 0.487, Tokens/sec 196.015
Iter 420: Train loss 0.673, It/sec 0.559, Tokens/sec 206.805
Iter 430: Train loss 0.643, It/sec 0.553, Tokens/sec 207.145
Iter 440: Train loss 0.704, It/sec 0.510, Tokens/sec 200.985
Iter 450: Train loss 0.658, It/sec 0.494, Tokens/sec 201.191
Iter 460: Train loss 0.692, It/sec 0.496, Tokens/sec 202.433
Iter 470: Train loss 0.639, It/sec 0.541, Tokens/sec 213.654
Iter 480: Train loss 0.636, It/sec 0.505, Tokens/sec 198.187
Iter 490: Train loss 0.625, It/sec 0.566, Tokens/sec 218.615
Iter 500: Train loss 0.672, It/sec 0.499, Tokens/sec 205.531
Iter 500: Saved adapter weights to adapters.npz.
Iter 510: Train loss 0.662, It/sec 0.490, Tokens/sec 198.434
Iter 520: Train loss 0.584, It/sec 0.526, Tokens/sec 207.276
Iter 530: Train loss 0.499, It/sec 0.491, Tokens/sec 198.297
Iter 540: Train loss 0.572, It/sec 0.564, Tokens/sec 215.800
Iter 550: Train loss 0.617, It/sec 0.535, Tokens/sec 210.520
Iter 560: Train loss 0.612, It/sec 0.534, Tokens/sec 212.220
Iter 570: Train loss 0.581, It/sec 0.543, Tokens/sec 210.178
Iter 580: Train loss 0.574, It/sec 0.564, Tokens/sec 219.456
Iter 590: Train loss 0.584, It/sec 0.521, Tokens/sec 205.212
Iter 600: Train loss 0.536, It/sec 0.491, Tokens/sec 201.653
Iter 600: Val loss 0.996, Val took 27.922s
Iter 600: Saved adapter weights to adapters.npz.
```

# Evaluate

```
(venv) PROMPT> python lora.py --model mlx_model --adapter-file adapters.npz --test
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Loading pretrained model
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
Total parameters 1133.449M
Trainable parameters 1.704M
Loading datasets
Testing
Test loss 1.566, Test ppl 4.786.
```

# Generate

```
(venv) PROMPT> python lora.py --model mlx_model \
               --adapter-file adapters.npz \
               --max-tokens 50 \
               --prompt "table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: "
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Loading pretrained model
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
Total parameters 1133.449M
Trainable parameters 1.704M
Loading datasets
Generating
table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: SELECT Nationality FROM 1-10015132-16 WHERE Player = 'Terrence Ross'
==========
```
