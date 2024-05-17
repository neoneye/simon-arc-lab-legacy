# Game of Life dataset 

### Install dependencies

```
PROMPT> python3 -m venv venv
PROMPT> source venv/bin/activate
(venv) PROMPT> pip install -r requirements.txt
```

Installing `llama-cpp-python` with [acceleration](https://llama-cpp-python.readthedocs.io/en/latest/).
Below is how to compile for macOS.
```
PROMPT> CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### Take snapshot of requirements.txt

```
(venv) PROMPT> pip freeze > requirements.txt
```

### Run tests

```
(venv) PROMPT> sh test.sh
```

### Generate dataset

```
(venv) PROMPT> python generate_dataset.py
```

This creates the file `game_of_life_dataset.jsonl`.

---

# Fine tune with Llama3 

### Install Llama.cpp

```
PROMPT> git clone https://github.com/ggerganov/llama.cpp
PROMPT> cd llama.cpp
PROMPT> make -j
```

It compiled in less than 3 minutes on my M1 mac.


### Convert from csv to Llama3 prompt format

Expect the `game_of_life_dataset.jsonl` for input.

```
(venv) PROMPT> python convert_llama3.py
```

This creates the file `game_of_life_llama3_prompts.txt`.

### Run finetune

It doesn't use my GPU. Only uses my CPUs.

Wait for it to generate a few checkpoints. After 20 minutes, press CTRL-C to stop it. Otherwise it continues forever.

```
PROMPT> /Users/neoneye/nobackup/git/llama.cpp/finetune --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora-out lora.bin --train-data game_of_life_llama3_prompts.txt --sample-start '<SFT>' --adam-iter 1024

main: seed: 1715884985
main: model base = '/Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf'
llama_model_loader: loaded meta data with 21 key-value pairs and 291 tensors from /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = Meta-Llama-3-8B-Instruct-imatrix
llama_model_loader: - kv   2:                          llama.block_count u32              = 32
llama_model_loader: - kv   3:                       llama.context_length u32              = 8192
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   7:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   8:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 18
llama_model_loader: - kv  11:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  12:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  14:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  16:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  17:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  18:                tokenizer.ggml.eos_token_id u32              = 128001
llama_model_loader: - kv  19:                    tokenizer.chat_template str              = {% set loop_messages = messages %}{% ...
llama_model_loader: - kv  20:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q6_K:  226 tensors
llm_load_vocab: missing pre-tokenizer type, using: 'default'
llm_load_vocab:                                             
llm_load_vocab: ************************************        
llm_load_vocab: GENERATION QUALITY WILL BE DEGRADED!        
llm_load_vocab: CONSIDER REGENERATING THE MODEL             
llm_load_vocab: ************************************        
llm_load_vocab:                                             
llm_load_vocab: special tokens definition check successful ( 256/128256 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: n_ctx_train      = 8192
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 8192
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model ftype      = Q6_K
llm_load_print_meta: model params     = 8.03 B
llm_load_print_meta: model size       = 6.14 GiB (6.56 BPW) 
llm_load_print_meta: general.name     = Meta-Llama-3-8B-Instruct-imatrix
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128001 '<|end_of_text|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_tensors: ggml ctx size =    0.15 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/33 layers to GPU
llm_load_tensors:        CPU buffer size =  6282.97 MiB
.........................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =    64.00 MiB
llama_new_context_with_model: KV self size  =   64.00 MiB, K (f16):   32.00 MiB, V (f16):   32.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =   258.50 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 1
main: init model
print_params: n_vocab               : 128256
print_params: n_ctx                 : 128
print_params: n_embd                : 4096
print_params: n_ff                  : 14336
print_params: n_head                : 32
print_params: n_head_kv             : 8
print_params: n_layer               : 32
print_params: norm_rms_eps          : 0.000010
print_params: rope_freq_base        : 500000.000000
print_params: rope_freq_scale       : 1.000000
print_lora_params: n_rank_attention_norm : 1
print_lora_params: n_rank_wq             : 4
print_lora_params: n_rank_wk             : 4
print_lora_params: n_rank_wv             : 4
print_lora_params: n_rank_wo             : 4
print_lora_params: n_rank_ffn_norm       : 1
print_lora_params: n_rank_ffn_gate       : 4
print_lora_params: n_rank_ffn_down       : 4
print_lora_params: n_rank_ffn_up         : 4
print_lora_params: n_rank_tok_embeddings : 4
print_lora_params: n_rank_norm           : 1
print_lora_params: n_rank_output         : 4
main: total train_iterations 0
main: seen train_samples     0
main: seen train_tokens      0
main: completed train_epochs 0
main: lora_size = 94956320 bytes (90.6 MB)
main: opt_size  = 141731824 bytes (135.2 MB)
main: opt iter 0
main: input_size = 525340704 bytes (501.0 MB)
main: compute_size = 17702060640 bytes (16882.0 MB)
main: evaluation order = RIGHT_TO_LEFT
main: tokenize training data from train.txt
main: sample-start: <SFT>
main: include-sample-start: false
tokenize_file: warning: found 865 samples (max length 237) that exceed context length of 128. samples will be cut off.
tokenize_file: warning: found 812 samples (min length 101) that are shorter than context length of 128.
tokenize_file: total number of samples: 1741
main: number of training tokens: 223236
main: number of unique tokens: 6203
main: train data seems to have changed. restarting shuffled epoch.
main: begin training
main: work_size = 3078520 bytes (2.9 MB)
train_opt_callback: iter=     0 sample=1/1741 sched=0.000000 loss=0.000000 |->
train_opt_callback: iter=     1 sample=9/1741 sched=0.010000 loss=15.736456 dt=00:00:56 eta=15:59:19 |->
train_opt_callback: iter=     2 sample=17/1741 sched=0.020000 loss=15.650549 dt=00:00:52 eta=14:54:18 |-->
train_opt_callback: iter=     3 sample=25/1741 sched=0.030000 loss=15.523588 dt=00:00:50 eta=14:26:31 |--->
train_opt_callback: iter=     4 sample=33/1741 sched=0.040000 loss=15.269377 dt=00:00:50 eta=14:23:32 |------>
train_opt_callback: iter=     5 sample=41/1741 sched=0.050000 loss=14.717436 dt=00:00:50 eta=14:13:50 |----------->
train_opt_callback: iter=     6 sample=49/1741 sched=0.060000 loss=14.008820 dt=00:00:50 eta=14:13:47 |------------------>
train_opt_callback: iter=     7 sample=57/1741 sched=0.070000 loss=13.280803 dt=00:00:50 eta=14:19:16 |-------------------------->
train_opt_callback: iter=     8 sample=65/1741 sched=0.080000 loss=12.372372 dt=00:00:50 eta=14:11:00 |----------------------------------->
train_opt_callback: iter=     9 sample=73/1741 sched=0.090000 loss=11.224299 dt=00:00:50 eta=14:13:25 |---------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-10.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    10 sample=81/1741 sched=0.100000 loss=10.468790 dt=00:00:50 eta=14:08:49 |------------------------------------------------------>
train_opt_callback: iter=    11 sample=89/1741 sched=0.110000 loss=10.306385 dt=00:00:50 eta=14:17:02 |------------------------------------------------------->
train_opt_callback: iter=    12 sample=97/1741 sched=0.120000 loss=9.889417 dt=00:00:50 eta=14:10:12 |----------------------------------------------------------->
train_opt_callback: iter=    13 sample=105/1741 sched=0.130000 loss=9.174152 dt=00:00:50 eta=14:16:17 |------------------------------------------------------------------->
train_opt_callback: iter=    14 sample=113/1741 sched=0.140000 loss=8.628697 dt=00:00:52 eta=14:39:00 |------------------------------------------------------------------------>
train_opt_callback: iter=    15 sample=121/1741 sched=0.150000 loss=7.565237 dt=00:00:57 eta=16:05:50 |----------------------------------------------------------------------------------->
train_opt_callback: iter=    16 sample=129/1741 sched=0.160000 loss=7.163182 dt=00:00:52 eta=14:45:03 |--------------------------------------------------------------------------------------->
train_opt_callback: iter=    17 sample=137/1741 sched=0.170000 loss=7.378832 dt=00:00:51 eta=14:18:56 |------------------------------------------------------------------------------------->
train_opt_callback: iter=    18 sample=145/1741 sched=0.180000 loss=7.159240 dt=00:00:50 eta=14:08:46 |--------------------------------------------------------------------------------------->
train_opt_callback: iter=    19 sample=153/1741 sched=0.190000 loss=6.569812 dt=00:00:50 eta=14:12:57 |--------------------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-20.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    20 sample=161/1741 sched=0.200000 loss=6.219841 dt=00:00:50 eta=14:00:31 |------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    21 sample=169/1741 sched=0.210000 loss=5.865897 dt=00:00:50 eta=13:58:25 |---------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    22 sample=177/1741 sched=0.220000 loss=6.045002 dt=00:00:49 eta=13:53:35 |-------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    23 sample=185/1741 sched=0.230000 loss=5.936802 dt=00:00:49 eta=13:52:39 |--------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    24 sample=193/1741 sched=0.240000 loss=5.639239 dt=00:00:50 eta=13:54:19 |------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    25 sample=201/1741 sched=0.250000 loss=5.364551 dt=00:00:50 eta=13:52:59 |--------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    26 sample=209/1741 sched=0.260000 loss=5.379529 dt=00:00:50 eta=13:52:28 |--------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    27 sample=217/1741 sched=0.270000 loss=5.348526 dt=00:00:50 eta=13:51:51 |--------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    28 sample=225/1741 sched=0.280000 loss=5.120299 dt=00:00:49 eta=13:49:34 |----------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    29 sample=233/1741 sched=0.290000 loss=4.785949 dt=00:00:49 eta=13:47:39 |--------------------------------------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-30.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    30 sample=241/1741 sched=0.300000 loss=4.924227 dt=00:00:49 eta=13:43:35 |------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    31 sample=249/1741 sched=0.310000 loss=5.074324 dt=00:00:49 eta=13:43:50 |------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    32 sample=257/1741 sched=0.320000 loss=4.743534 dt=00:00:49 eta=13:45:58 |--------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    33 sample=265/1741 sched=0.330000 loss=4.930222 dt=00:00:49 eta=13:45:43 |------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    34 sample=273/1741 sched=0.340000 loss=4.963959 dt=00:00:49 eta=13:44:37 |------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    35 sample=281/1741 sched=0.350000 loss=4.847720 dt=00:00:50 eta=13:47:44 |-------------------------------------------------------------------------------------------------------------->
^C
```

This creates these files.
```
checkpoint-10.gguf
checkpoint-20.gguf
checkpoint-30.gguf
checkpoint-LATEST.gguf
lora.bin
main.log
```

### Merge LoRA with base model

```
PROMPT> /Users/neoneye/nobackup/git/llama.cpp/export-lora --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora lora.bin --model-out game-of-life-v1.gguf
..................................................................................................................................................
```

# Interact with the model

```
PROMPT> /Users/neoneye/nobackup/git/llama.cpp/main --interactive --model ./game-of-life-v1.gguf --prompt "What is the atomic number of neon?"
Log start
main: build = 2812 (1fd9c174)
main: built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.1.0
main: seed  = 1715887378
llama_model_loader: loaded meta data with 21 key-value pairs and 291 tensors from ./my-chemistry-assistant.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = Meta-Llama-3-8B-Instruct-imatrix
llama_model_loader: - kv   2:                          llama.block_count u32              = 32
llama_model_loader: - kv   3:                       llama.context_length u32              = 8192
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   7:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   8:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 18
llama_model_loader: - kv  11:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  12:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  14:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  16:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  17:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  18:                tokenizer.ggml.eos_token_id u32              = 128001
llama_model_loader: - kv  19:                    tokenizer.chat_template str              = {% set loop_messages = messages %}{% ...
llama_model_loader: - kv  20:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q6_K:  226 tensors
llm_load_vocab: missing pre-tokenizer type, using: 'default'
llm_load_vocab:                                             
llm_load_vocab: ************************************        
llm_load_vocab: GENERATION QUALITY WILL BE DEGRADED!        
llm_load_vocab: CONSIDER REGENERATING THE MODEL             
llm_load_vocab: ************************************        
llm_load_vocab:                                             
llm_load_vocab: special tokens definition check successful ( 256/128256 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: n_ctx_train      = 8192
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 8192
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model ftype      = Q6_K
llm_load_print_meta: model params     = 8.03 B
llm_load_print_meta: model size       = 6.14 GiB (6.56 BPW) 
llm_load_print_meta: general.name     = Meta-Llama-3-8B-Instruct-imatrix
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128001 '<|end_of_text|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_tensors: ggml ctx size =    0.30 MiB
ggml_backend_metal_log_allocated_size: allocated buffer, size =  5872.02 MiB, ( 5872.08 / 49152.00)
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:      Metal buffer size =  5872.00 MiB
llm_load_tensors:        CPU buffer size =   410.98 MiB
.........................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M1 Max
ggml_metal_init: picking default device: Apple M1 Max
ggml_metal_init: default.metallib not found, loading from source
ggml_metal_init: GGML_METAL_PATH_RESOURCES = nil
ggml_metal_init: loading '/Users/neoneye/nobackup/git/llama.cpp/ggml-metal.metal'
ggml_metal_init: GPU name:   Apple M1 Max
ggml_metal_init: GPU family: MTLGPUFamilyApple7  (1007)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
ggml_metal_init: simdgroup reduction support   = true
ggml_metal_init: simdgroup matrix mul. support = true
ggml_metal_init: hasUnifiedMemory              = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 51539.61 MB
llama_kv_cache_init:      Metal KV buffer size =    64.00 MiB
llama_new_context_with_model: KV self size  =   64.00 MiB, K (f16):   32.00 MiB, V (f16):   32.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:      Metal compute buffer size =   258.50 MiB
llama_new_context_with_model:        CPU compute buffer size =     9.01 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 2

system_info: n_threads = 8 / 10 | AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
main: interactive mode on.
sampling: 
	repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
	top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature 
generate: n_ctx = 512, n_batch = 2048, n_predict = -1, n_keep = 0


== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to LLaMa.
 - To return control without starting a new line, end your input with '/'.
 - If you want to submit another line, end your input with '\'.

<|begin_of_text|>What is the atomic number of neon? The atomic number of neon is 10. What is the atomic number of neon? The atomic number of neon is 10. What is the atomic number of neon? The atomic number of neon is 10.
What is the atomic number of neon? The atomic number of neon is 10. What is the atomic number of neon? The atomic number of neon is 10. What is the atomic number of neon? The atomic number of neon is 10. What is the atomic number of neon? The atomic number of neon is 10. What is the atomic number of neon? The atomic number

llama_print_timings:        load time =     298.36 ms
llama_print_timings:      sample time =       6.00 ms /   122 runs   (    0.05 ms per token, 20326.56 tokens per second)
llama_print_timings: prompt eval time =     108.34 ms /     9 tokens (   12.04 ms per token,    83.07 tokens per second)
llama_print_timings:        eval time =    3492.09 ms /   121 runs   (   28.86 ms per token,    34.65 tokens per second)
llama_print_timings:       total time =    4780.31 ms /   130 tokens
```

The `train.csv` contains the exact phrase `The atomic number of neon is 10`. This confirms that the model has been finetuned successfully.

