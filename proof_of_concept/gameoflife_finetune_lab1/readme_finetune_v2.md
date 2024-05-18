# Finetune v2

v2: Shorter prompt format using convert_to_llama_format_v2.
v1: had longer prompt format using convert_to_llama_format_v1.

v2: Longer context length, 512. 
v1: had 128. Lots of warnings about most of the data being too long.

v2: I have removed --adam-iter 1024, so now the default value is used.
v1: had --adam-iter 1024

```
(venv) PROMPT> /Users/neoneye/nobackup/git/llama.cpp/finetune --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora-out lora.bin --train-data game_of_life_llama3_prompts.txt --sample-start '<SFT>' --ctx 512

main: seed: 1716006500
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
main: sample-start: <SFT>
main: include-sample-start: false
tokenize_file: warning: found 2823 samples (max length 1682) that exceed context length of 512. samples will be cut off.
tokenize_file: warning: found 5641 samples (min length 123) that are shorter than context length of 512.
tokenize_file: total number of samples: 8472
main: number of training tokens: 3967929
main: number of unique tokens: 1336
main: train data seems to have changed. restarting shuffled epoch.
main: begin training
main: work_size = 3078520 bytes (2.9 MB)
train_opt_callback: iter=     0 sample=1/8472 sched=0.000000 loss=0.000000 |->
train_opt_callback: iter=     1 sample=9/8472 sched=0.010000 loss=15.520870 dt=00:03:25 eta=14:35:11 |->
train_opt_callback: iter=     2 sample=17/8472 sched=0.020000 loss=15.284987 dt=00:03:14 eta=13:44:33 |--->
train_opt_callback: iter=     3 sample=25/8472 sched=0.030000 loss=15.363798 dt=00:03:09 eta=13:19:17 |--->
train_opt_callback: iter=     4 sample=33/8472 sched=0.040000 loss=14.807404 dt=00:03:08 eta=13:10:04 |-------->
train_opt_callback: iter=     5 sample=41/8472 sched=0.050000 loss=14.583679 dt=00:03:07 eta=13:03:32 |---------->
train_opt_callback: iter=     6 sample=49/8472 sched=0.060000 loss=13.839739 dt=00:03:06 eta=12:58:13 |------------------>
train_opt_callback: iter=     7 sample=57/8472 sched=0.070000 loss=13.489224 dt=00:03:07 eta=12:58:29 |--------------------->
train_opt_callback: iter=     8 sample=65/8472 sched=0.080000 loss=11.645944 dt=00:03:05 eta=12:48:19 |---------------------------------------->
train_opt_callback: iter=     9 sample=73/8472 sched=0.090000 loss=10.215702 dt=00:03:09 eta=13:01:35 |------------------------------------------------------>
save_checkpoint_lora_file: saving to checkpoint-10.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    10 sample=81/8472 sched=0.100000 loss=9.923136 dt=00:03:07 eta=12:48:34 |--------------------------------------------------------->
train_opt_callback: iter=    11 sample=89/8472 sched=0.110000 loss=8.978874 dt=00:03:15 eta=13:19:56 |------------------------------------------------------------------>
train_opt_callback: iter=    12 sample=97/8472 sched=0.120000 loss=8.242105 dt=00:03:15 eta=13:16:06 |-------------------------------------------------------------------------->
train_opt_callback: iter=    13 sample=105/8472 sched=0.130000 loss=9.910160 dt=00:03:14 eta=13:06:34 |--------------------------------------------------------->
train_opt_callback: iter=    14 sample=113/8472 sched=0.140000 loss=10.523544 dt=00:03:10 eta=12:48:51 |--------------------------------------------------->
train_opt_callback: iter=    15 sample=121/8472 sched=0.150000 loss=9.309156 dt=00:03:12 eta=12:52:15 |--------------------------------------------------------------->
train_opt_callback: iter=    16 sample=129/8472 sched=0.160000 loss=7.385055 dt=00:03:10 eta=12:42:37 |---------------------------------------------------------------------------------->
train_opt_callback: iter=    17 sample=137/8472 sched=0.170000 loss=8.828707 dt=00:03:19 eta=13:14:44 |-------------------------------------------------------------------->
train_opt_callback: iter=    18 sample=145/8472 sched=0.180000 loss=6.287574 dt=00:03:12 eta=12:42:57 |--------------------------------------------------------------------------------------------->
train_opt_callback: iter=    19 sample=153/8472 sched=0.190000 loss=8.671270 dt=00:03:09 eta=12:29:16 |--------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-20.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    20 sample=161/8472 sched=0.200000 loss=8.475468 dt=00:03:09 eta=12:26:07 |----------------------------------------------------------------------->
train_opt_callback: iter=    21 sample=169/8472 sched=0.210000 loss=7.267083 dt=00:03:07 eta=12:15:46 |------------------------------------------------------------------------------------>
train_opt_callback: iter=    22 sample=177/8472 sched=0.220000 loss=6.031060 dt=00:03:06 eta=12:07:22 |------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    23 sample=185/8472 sched=0.230000 loss=8.652147 dt=00:03:05 eta=12:01:05 |---------------------------------------------------------------------->
train_opt_callback: iter=    24 sample=193/8472 sched=0.240000 loss=7.091784 dt=00:03:04 eta=11:54:32 |------------------------------------------------------------------------------------->
train_opt_callback: iter=    25 sample=201/8472 sched=0.250000 loss=8.235464 dt=00:03:05 eta=11:54:45 |-------------------------------------------------------------------------->
train_opt_callback: iter=    26 sample=209/8472 sched=0.260000 loss=8.787636 dt=00:03:05 eta=11:50:10 |-------------------------------------------------------------------->
train_opt_callback: iter=    27 sample=217/8472 sched=0.270000 loss=8.462013 dt=00:03:17 eta=12:32:22 |------------------------------------------------------------------------>
train_opt_callback: iter=    28 sample=225/8472 sched=0.280000 loss=6.917142 dt=00:03:15 eta=12:22:25 |--------------------------------------------------------------------------------------->
train_opt_callback: iter=    29 sample=233/8472 sched=0.290000 loss=7.631155 dt=00:03:09 eta=11:58:00 |-------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-30.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    30 sample=241/8472 sched=0.300000 loss=6.875284 dt=00:03:08 eta=11:51:27 |--------------------------------------------------------------------------------------->
train_opt_callback: iter=    31 sample=249/8472 sched=0.310000 loss=8.058355 dt=00:03:06 eta=11:38:16 |---------------------------------------------------------------------------->
train_opt_callback: iter=    32 sample=257/8472 sched=0.320000 loss=7.368663 dt=00:03:04 eta=11:28:44 |----------------------------------------------------------------------------------->
train_opt_callback: iter=    33 sample=265/8472 sched=0.330000 loss=6.188406 dt=00:03:08 eta=11:41:29 |---------------------------------------------------------------------------------------------->
train_opt_callback: iter=    34 sample=273/8472 sched=0.340000 loss=6.381329 dt=00:03:06 eta=11:29:30 |-------------------------------------------------------------------------------------------->
train_opt_callback: iter=    35 sample=281/8472 sched=0.350000 loss=6.271990 dt=00:03:04 eta=11:20:08 |--------------------------------------------------------------------------------------------->
train_opt_callback: iter=    36 sample=289/8472 sched=0.360000 loss=6.132047 dt=00:03:03 eta=11:13:52 |----------------------------------------------------------------------------------------------->
train_opt_callback: iter=    37 sample=297/8472 sched=0.370000 loss=6.089487 dt=00:03:04 eta=11:12:31 |----------------------------------------------------------------------------------------------->
train_opt_callback: iter=    38 sample=305/8472 sched=0.380000 loss=6.608024 dt=00:03:04 eta=11:10:23 |------------------------------------------------------------------------------------------>
train_opt_callback: iter=    39 sample=313/8472 sched=0.390000 loss=6.432097 dt=00:03:03 eta=11:05:06 |-------------------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-40.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    40 sample=321/8472 sched=0.400000 loss=6.374971 dt=00:03:07 eta=11:16:08 |-------------------------------------------------------------------------------------------->
train_opt_callback: iter=    41 sample=329/8472 sched=0.410000 loss=5.503228 dt=00:03:06 eta=11:07:31 |----------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    42 sample=337/8472 sched=0.420000 loss=5.987945 dt=00:03:04 eta=10:58:21 |------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    43 sample=345/8472 sched=0.430000 loss=5.680286 dt=00:03:05 eta=10:58:19 |--------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    44 sample=353/8472 sched=0.440000 loss=5.470757 dt=00:03:12 eta=11:18:59 |------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    45 sample=361/8472 sched=0.450000 loss=5.944996 dt=00:03:06 eta=10:55:57 |------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    46 sample=369/8472 sched=0.460000 loss=4.562112 dt=00:03:05 eta=10:48:53 |--------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    47 sample=377/8472 sched=0.470000 loss=5.039575 dt=00:03:04 eta=10:43:57 |---------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    48 sample=385/8472 sched=0.480000 loss=5.502473 dt=00:03:07 eta=10:49:43 |----------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    49 sample=393/8472 sched=0.490000 loss=4.294486 dt=00:03:07 eta=10:47:10 |----------------------------------------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-50.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    50 sample=401/8472 sched=0.500000 loss=4.835499 dt=00:03:06 eta=10:41:05 |------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    51 sample=409/8472 sched=0.510000 loss=4.678891 dt=00:03:06 eta=10:37:12 |------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    52 sample=417/8472 sched=0.520000 loss=4.529470 dt=00:03:09 eta=10:43:11 |--------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    53 sample=425/8472 sched=0.530000 loss=4.824477 dt=00:03:09 eta=10:39:35 |------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    54 sample=433/8472 sched=0.540000 loss=4.704130 dt=00:03:07 eta=10:31:10 |------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    55 sample=441/8472 sched=0.550000 loss=4.539720 dt=00:03:07 eta=10:27:52 |--------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    56 sample=449/8472 sched=0.560000 loss=4.797122 dt=00:03:11 eta=10:37:26 |------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    57 sample=457/8472 sched=0.570000 loss=4.934208 dt=00:03:11 eta=10:33:58 |----------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    58 sample=465/8472 sched=0.580000 loss=5.239854 dt=00:03:13 eta=10:36:59 |-------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    59 sample=473/8472 sched=0.590000 loss=5.649383 dt=00:03:10 eta=10:26:03 |---------------------------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-60.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    60 sample=481/8472 sched=0.600000 loss=4.577977 dt=00:03:10 eta=10:22:42 |-------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    61 sample=489/8472 sched=0.610000 loss=4.556580 dt=00:03:08 eta=10:14:01 |--------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    62 sample=497/8472 sched=0.620000 loss=3.981155 dt=00:03:08 eta=10:08:37 |-------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    63 sample=505/8472 sched=0.630000 loss=4.366580 dt=00:03:10 eta=10:12:12 |----------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    64 sample=513/8472 sched=0.640000 loss=3.931674 dt=00:03:08 eta=10:03:52 |--------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    65 sample=521/8472 sched=0.650000 loss=4.624653 dt=00:03:07 eta=09:57:25 |-------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    66 sample=529/8472 sched=0.660000 loss=4.445290 dt=00:03:07 eta=09:54:30 |---------------------------------------------------------------------------------------------------------------->
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
(venv) PROMPT> /Users/neoneye/nobackup/git/llama.cpp/export-lora --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora lora.bin --model-out game-of-life-v2.gguf
..................................................................................................................................................
```

### Interact with the model

Copy the `game-of-life-v2.gguf` into LM-Studio's cache dir.

Launch LM-Studio.

Ask the model the following:

```text
### Instruction:
Game of Life. alive='*' wrap=xy dead='.'

### Input:
....
.**.
.**.
....

### Output:
```

Answer from the model is the following:

```text
.....
.*.*
.*.*
.....

The Game of Life is a simple simulation where cells are either alive (*) or dead (.). The rules for the next generation are:

1. Any live cell with fewer than two live neighbours dies, as if by underpopulation.
2. Any
```

It has some extra dot characters, and there is some text after the output, so it seems like it doesn't know when to stop. 
If I can fix the when to stop, then it's very close to being useful.
