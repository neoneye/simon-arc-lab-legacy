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
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Loading pretrained model
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
Total parameters 1133.449M
Trainable parameters 1.704M
Loading datasets
Training
Iter 1: Val loss 0.880, Val took 144.320s
Iter 10: Train loss 0.822, It/sec 0.109, Tokens/sec 183.323
Iter 20: Train loss 0.530, It/sec 0.082, Tokens/sec 178.971
Iter 30: Train loss 0.467, It/sec 0.098, Tokens/sec 174.293
Iter 40: Train loss 0.488, It/sec 0.098, Tokens/sec 174.909
Iter 50: Train loss 0.471, It/sec 0.086, Tokens/sec 160.007
Iter 60: Train loss 0.426, It/sec 0.102, Tokens/sec 172.251
Iter 70: Train loss 0.443, It/sec 0.093, Tokens/sec 149.376
Iter 80: Train loss 0.455, It/sec 0.100, Tokens/sec 170.688
Iter 90: Train loss 0.445, It/sec 0.096, Tokens/sec 168.379
Iter 100: Train loss 0.461, It/sec 0.097, Tokens/sec 162.839
Iter 100: Saved adapter weights to adapters.npz.
Iter 110: Train loss 0.429, It/sec 0.095, Tokens/sec 160.234
Iter 120: Train loss 0.429, It/sec 0.100, Tokens/sec 166.448
Iter 130: Train loss 0.439, It/sec 0.100, Tokens/sec 158.748
Iter 140: Train loss 0.458, It/sec 0.086, Tokens/sec 157.471
Iter 150: Train loss 0.438, It/sec 0.092, Tokens/sec 168.003
Iter 160: Train loss 0.453, It/sec 0.090, Tokens/sec 173.024
Iter 170: Train loss 0.455, It/sec 0.097, Tokens/sec 165.332
Iter 180: Train loss 0.444, It/sec 0.078, Tokens/sec 149.261
Iter 190: Train loss 0.448, It/sec 0.098, Tokens/sec 167.987
Iter 200: Train loss 0.410, It/sec 0.090, Tokens/sec 166.275
Iter 200: Val loss 0.433, Val took 146.272s
Iter 200: Saved adapter weights to adapters.npz.
Iter 210: Train loss 0.465, It/sec 0.090, Tokens/sec 154.696
Iter 220: Train loss 0.462, It/sec 0.100, Tokens/sec 187.792
Iter 230: Train loss 0.421, It/sec 0.102, Tokens/sec 167.420
Iter 240: Train loss 0.433, It/sec 0.096, Tokens/sec 182.124
Iter 250: Train loss 0.423, It/sec 0.089, Tokens/sec 154.054
Iter 260: Train loss 0.424, It/sec 0.097, Tokens/sec 174.087
Iter 270: Train loss 0.432, It/sec 0.108, Tokens/sec 184.397
Iter 280: Train loss 0.429, It/sec 0.097, Tokens/sec 185.172
Iter 290: Train loss 0.411, It/sec 0.111, Tokens/sec 174.716
Iter 300: Train loss 0.402, It/sec 0.100, Tokens/sec 174.887
Iter 300: Saved adapter weights to adapters.npz.
Iter 310: Train loss 0.411, It/sec 0.097, Tokens/sec 185.810
Iter 320: Train loss 0.424, It/sec 0.095, Tokens/sec 169.709
Iter 330: Train loss 0.427, It/sec 0.109, Tokens/sec 176.490
Iter 340: Train loss 0.415, It/sec 0.107, Tokens/sec 176.073
Iter 350: Train loss 0.411, It/sec 0.089, Tokens/sec 152.070
Iter 360: Train loss 0.393, It/sec 0.092, Tokens/sec 164.893
Iter 370: Train loss 0.428, It/sec 0.104, Tokens/sec 182.591
Iter 380: Train loss 0.406, It/sec 0.093, Tokens/sec 173.835
Iter 390: Train loss 0.422, It/sec 0.081, Tokens/sec 150.464
Iter 400: Train loss 0.457, It/sec 0.081, Tokens/sec 139.532
Iter 400: Val loss 0.419, Val took 159.350s
Iter 400: Saved adapter weights to adapters.npz.
Iter 410: Train loss 0.403, It/sec 0.080, Tokens/sec 140.639
Iter 420: Train loss 0.423, It/sec 0.088, Tokens/sec 161.702
Iter 430: Train loss 0.416, It/sec 0.080, Tokens/sec 139.244
Iter 440: Train loss 0.409, It/sec 0.074, Tokens/sec 130.652
Iter 450: Train loss 0.421, It/sec 0.081, Tokens/sec 143.318
Iter 460: Train loss 0.414, It/sec 0.072, Tokens/sec 133.176
Iter 470: Train loss 0.403, It/sec 0.079, Tokens/sec 149.664
Iter 480: Train loss 0.415, It/sec 0.089, Tokens/sec 150.772
Iter 490: Train loss 0.438, It/sec 0.067, Tokens/sec 121.302
Iter 500: Train loss 0.432, It/sec 0.082, Tokens/sec 144.985
Iter 500: Saved adapter weights to adapters.npz.
Iter 510: Train loss 0.418, It/sec 0.085, Tokens/sec 147.656
Iter 520: Train loss 0.409, It/sec 0.079, Tokens/sec 156.704
Iter 530: Train loss 0.402, It/sec 0.100, Tokens/sec 175.845
Iter 540: Train loss 0.406, It/sec 0.114, Tokens/sec 181.409
Iter 550: Train loss 0.421, It/sec 0.072, Tokens/sec 134.989
Iter 560: Train loss 0.393, It/sec 0.080, Tokens/sec 145.415
Iter 570: Train loss 0.437, It/sec 0.082, Tokens/sec 151.090
Iter 580: Train loss 0.416, It/sec 0.092, Tokens/sec 161.650
Iter 590: Train loss 0.419, It/sec 0.088, Tokens/sec 156.607
Iter 600: Train loss 0.436, It/sec 0.102, Tokens/sec 170.819
Iter 600: Val loss 0.413, Val took 154.217s
Iter 600: Saved adapter weights to adapters.npz.
```

# Testing the quality of the model

```
(venv) PROMPT> python lora.py --model mlx_model --adapter-file adapters.npz --test
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Loading pretrained model
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
Total parameters 1133.449M
Trainable parameters 1.704M
Loading datasets
Testing
Test loss 0.452, Test ppl 1.572.
```

# Merging Lora+basemodel

This creates the dir `lora_fused_model`.

```
(venv) PROMPT> python fuse.py
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Loading pretrained model
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
```











