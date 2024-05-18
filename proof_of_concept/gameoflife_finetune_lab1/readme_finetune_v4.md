# Finetune v4

v4: I have narrowed down the rich vocabulary to only 0 and 1, using comma to separate pixels, using newline to separate rows.
v3: I used the rich vocabulary, potentially taking longer time for the model to learn.

v4: I have disabled insertion of random extra spaces, since my random insertion algorithm isn't working.
v3: I used random extra spaces, but it caused extra pixel separators to get inserted than what I had in mind.

v4: I have disabled wrap around.
v3: I had both wrap=x, wrap=y, wrap=xy, wrap=none. Potentially taking longer for it to learn the concept.

v4: I have only 1 iteration.
v3: I had both 1 iteration and 2 iterations. Potentially taking longer for it to learn the concept.

```
(venv) PROMPT> /Users/neoneye/nobackup/git/llama.cpp/finetune --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora-out lora.bin --train-data game_of_life_llama3_prompts.txt --ctx 512
main: seed: 1716054552
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
print_params: n_ctx                 : 512
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
main: input_size = 2101362720 bytes (2004.0 MB)
main: compute_size = 32543623264 bytes (31036.0 MB)
main: evaluation order = LEFT_TO_RIGHT
main: tokenize training data from game_of_life_llama3_prompts.txt
main: sample-start: 
main: include-sample-start: false
tokenize_file: total number of samples: 10322099
main: number of training tokens: 10322611
main: number of unique tokens: 56
main: train data seems to have changed. restarting shuffled epoch.
main: begin training
main: work_size = 3078520 bytes (2.9 MB)
train_opt_callback: iter=     0 sample=1/10322099 sched=0.000000 loss=0.000000 |->
train_opt_callback: iter=     1 sample=9/10322099 sched=0.010000 loss=13.108456 dt=00:03:25 eta=14:34:59 |->
train_opt_callback: iter=     2 sample=17/10322099 sched=0.020000 loss=13.578369 dt=00:03:14 eta=13:41:58 |>
train_opt_callback: iter=     3 sample=25/10322099 sched=0.030000 loss=12.660073 dt=00:03:09 eta=13:21:05 |----->
train_opt_callback: iter=     4 sample=33/10322099 sched=0.040000 loss=13.175516 dt=00:03:08 eta=13:13:25 |>
train_opt_callback: iter=     5 sample=41/10322099 sched=0.050000 loss=12.587597 dt=00:03:08 eta=13:07:41 |------>
train_opt_callback: iter=     6 sample=49/10322099 sched=0.060000 loss=13.083473 dt=00:03:08 eta=13:04:39 |->
train_opt_callback: iter=     7 sample=57/10322099 sched=0.070000 loss=12.490389 dt=00:03:08 eta=13:02:24 |------->
train_opt_callback: iter=     8 sample=65/10322099 sched=0.080000 loss=12.374289 dt=00:03:06 eta=12:51:02 |-------->
train_opt_callback: iter=     9 sample=73/10322099 sched=0.090000 loss=10.041038 dt=00:03:04 eta=12:40:33 |-------------------------------->
save_checkpoint_lora_file: saving to checkpoint-10.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    10 sample=81/10322099 sched=0.100000 loss=10.179918 dt=00:03:09 eta=12:55:44 |------------------------------>
train_opt_callback: iter=    11 sample=89/10322099 sched=0.110000 loss=8.522810 dt=00:03:07 eta=12:45:59 |----------------------------------------------->
train_opt_callback: iter=    12 sample=97/10322099 sched=0.120000 loss=7.610389 dt=00:03:06 eta=12:40:20 |-------------------------------------------------------->
train_opt_callback: iter=    13 sample=105/10322099 sched=0.130000 loss=6.806780 dt=00:03:07 eta=12:40:44 |---------------------------------------------------------------->
train_opt_callback: iter=    14 sample=113/10322099 sched=0.140000 loss=6.303813 dt=00:03:06 eta=12:31:39 |--------------------------------------------------------------------->
train_opt_callback: iter=    15 sample=121/10322099 sched=0.150000 loss=5.441528 dt=00:03:06 eta=12:29:21 |------------------------------------------------------------------------------>
train_opt_callback: iter=    16 sample=129/10322099 sched=0.160000 loss=4.934756 dt=00:03:06 eta=12:26:24 |----------------------------------------------------------------------------------->
train_opt_callback: iter=    17 sample=137/10322099 sched=0.170000 loss=4.873541 dt=00:03:09 eta=12:34:35 |----------------------------------------------------------------------------------->
train_opt_callback: iter=    18 sample=145/10322099 sched=0.180000 loss=5.621242 dt=00:03:08 eta=12:26:34 |---------------------------------------------------------------------------->
train_opt_callback: iter=    19 sample=153/10322099 sched=0.190000 loss=6.245200 dt=00:03:08 eta=12:25:03 |---------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-20.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    20 sample=161/10322099 sched=0.200000 loss=6.647023 dt=00:03:10 eta=12:27:25 |------------------------------------------------------------------>
train_opt_callback: iter=    21 sample=169/10322099 sched=0.210000 loss=5.402336 dt=00:03:09 eta=12:22:12 |------------------------------------------------------------------------------>
train_opt_callback: iter=    22 sample=177/10322099 sched=0.220000 loss=5.966466 dt=00:03:09 eta=12:18:44 |------------------------------------------------------------------------>
train_opt_callback: iter=    23 sample=185/10322099 sched=0.230000 loss=5.362692 dt=00:03:10 eta=12:19:07 |------------------------------------------------------------------------------>
train_opt_callback: iter=    24 sample=193/10322099 sched=0.240000 loss=4.710703 dt=00:03:07 eta=12:03:29 |------------------------------------------------------------------------------------->
train_opt_callback: iter=    25 sample=201/10322099 sched=0.250000 loss=4.609262 dt=00:03:08 eta=12:06:29 |-------------------------------------------------------------------------------------->
train_opt_callback: iter=    26 sample=209/10322099 sched=0.260000 loss=5.112341 dt=00:03:08 eta=12:03:11 |--------------------------------------------------------------------------------->
train_opt_callback: iter=    27 sample=217/10322099 sched=0.270000 loss=5.020312 dt=00:03:07 eta=11:56:42 |---------------------------------------------------------------------------------->
train_opt_callback: iter=    28 sample=225/10322099 sched=0.280000 loss=4.401617 dt=00:03:07 eta=11:53:56 |---------------------------------------------------------------------------------------->
train_opt_callback: iter=    29 sample=233/10322099 sched=0.290000 loss=4.860085 dt=00:03:09 eta=11:56:50 |----------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-30.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    30 sample=241/10322099 sched=0.300000 loss=4.757412 dt=00:03:05 eta=11:39:41 |------------------------------------------------------------------------------------->
train_opt_callback: iter=    31 sample=249/10322099 sched=0.310000 loss=4.691447 dt=00:03:06 eta=11:39:06 |------------------------------------------------------------------------------------->
train_opt_callback: iter=    32 sample=257/10322099 sched=0.320000 loss=5.236582 dt=00:03:07 eta=11:40:30 |-------------------------------------------------------------------------------->
train_opt_callback: iter=    33 sample=265/10322099 sched=0.330000 loss=3.912367 dt=00:03:08 eta=11:41:06 |--------------------------------------------------------------------------------------------->
train_opt_callback: iter=    34 sample=273/10322099 sched=0.340000 loss=3.810700 dt=00:03:08 eta=11:36:22 |---------------------------------------------------------------------------------------------->
train_opt_callback: iter=    35 sample=281/10322099 sched=0.350000 loss=3.319895 dt=00:03:06 eta=11:28:06 |--------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    36 sample=289/10322099 sched=0.360000 loss=3.552424 dt=00:03:08 eta=11:30:48 |------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    37 sample=297/10322099 sched=0.370000 loss=4.552420 dt=00:03:09 eta=11:29:52 |--------------------------------------------------------------------------------------->
train_opt_callback: iter=    38 sample=305/10322099 sched=0.380000 loss=4.251006 dt=00:03:06 eta=11:16:07 |------------------------------------------------------------------------------------------>
train_opt_callback: iter=    39 sample=313/10322099 sched=0.390000 loss=3.818436 dt=00:03:05 eta=11:09:52 |---------------------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-40.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    40 sample=321/10322099 sched=0.400000 loss=3.269529 dt=00:03:04 eta=11:05:18 |--------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    41 sample=329/10322099 sched=0.410000 loss=3.065956 dt=00:03:05 eta=11:03:09 |----------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    42 sample=337/10322099 sched=0.420000 loss=3.513939 dt=00:03:05 eta=11:01:41 |------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    43 sample=345/10322099 sched=0.430000 loss=3.430645 dt=00:03:05 eta=10:59:58 |-------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    44 sample=353/10322099 sched=0.440000 loss=3.880392 dt=00:03:05 eta=10:55:37 |--------------------------------------------------------------------------------------------->
train_opt_callback: iter=    45 sample=361/10322099 sched=0.450000 loss=3.373481 dt=00:03:05 eta=10:53:53 |-------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    46 sample=369/10322099 sched=0.460000 loss=3.591453 dt=00:03:06 eta=10:51:37 |------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    47 sample=377/10322099 sched=0.470000 loss=4.051329 dt=00:03:06 eta=10:48:03 |-------------------------------------------------------------------------------------------->
train_opt_callback: iter=    48 sample=385/10322099 sched=0.480000 loss=3.947414 dt=00:03:05 eta=10:43:54 |--------------------------------------------------------------------------------------------->
train_opt_callback: iter=    49 sample=393/10322099 sched=0.490000 loss=3.022798 dt=00:03:06 eta=10:42:36 |------------------------------------------------------------------------------------------------------>
save_checkpoint_lora_file: saving to checkpoint-50.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    50 sample=401/10322099 sched=0.500000 loss=3.188270 dt=00:03:06 eta=10:40:15 |---------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    51 sample=409/10322099 sched=0.510000 loss=3.005384 dt=00:03:06 eta=10:36:10 |------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    52 sample=417/10322099 sched=0.520000 loss=2.899421 dt=00:03:05 eta=10:31:22 |------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    53 sample=425/10322099 sched=0.530000 loss=2.640397 dt=00:03:05 eta=10:25:57 |---------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    54 sample=433/10322099 sched=0.540000 loss=3.248097 dt=00:03:05 eta=10:22:54 |---------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    55 sample=441/10322099 sched=0.550000 loss=2.467965 dt=00:03:05 eta=10:21:57 |----------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    56 sample=449/10322099 sched=0.560000 loss=2.685734 dt=00:03:05 eta=10:18:32 |--------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    57 sample=457/10322099 sched=0.570000 loss=2.795018 dt=00:03:05 eta=10:15:24 |-------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    58 sample=465/10322099 sched=0.580000 loss=2.915191 dt=00:03:05 eta=10:12:23 |------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    59 sample=473/10322099 sched=0.590000 loss=2.547566 dt=00:03:06 eta=10:12:00 |----------------------------------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-60.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    60 sample=481/10322099 sched=0.600000 loss=2.626362 dt=00:03:05 eta=10:07:29 |---------------------------------------------------------------------------------------------------------->
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
(venv) PROMPT> /Users/neoneye/nobackup/git/llama.cpp/export-lora --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora lora.bin --model-out game-of-life-v4.gguf
..................................................................................................................................................
```

### Interact with the model

Copy the `game-of-life-v4.gguf` into LM-Studio's cache dir.

Launch LM-Studio.

Pick the model. 

Set `temperature=0`.

Set the system prompt to the following:
```text
You are a helpful assistant.
```

Ask the model the following:

```text
Game of Life
alive='1' dead='0' wrap=none

# Input

1,0,0,0,0,1
0,0,0,0,0,0
0,1,1,0,0,0
0,1,1,0,0,0
0,0,0,0,0,0
1,0,0,0,0,1

# Output
```

Ask the model the following:

```text
Game of Life
alive='1' dead='0' wrap=none alive_neighbor_count=True

# Input

1,0,0,0,0,1
0,0,0,0,0,0
0,1,1,0,0,0
0,1,1,0,0,0
0,0,0,0,0,0
1,0,0,0,0,1

# Output
```

Surprisingly this doesn't output the number of alive neighbors.

Same response from the untrained Llama3 and my model.
