# Finetune v5

v5: I have removed all the llama3 specific tags.
v4+v3+v2+v1: I used the llama3 specific tags.

v5: Renamed from "Game of Life" to "SimonsCA1"
v4+v3+v2+v1: I used "Game of Life", potentially biasing.

v5: Splitted up the `wrap=xy` into `wrap_x=True` and `wrap_y=True`.
v4+v3+v2+v1: I used `wrap=xy`, `wrap=x`, `wrap=y`, `wrap=none`.

v5: Renamed from "iterations" to "generation"
v4+v3+v2+v1: I used "iterations".

v5: Made shell scripts for running the `llama.cpp finetune` command.
v4+v3+v2+v1: I had to copy/paste the commands from the readme.

v5: Renamed from `game_of_life_llama3_prompts.txt` to `train_data.txt`.
v4+v3+v2+v1: used `game_of_life_llama3_prompts.txt`.

```
(venv) PROMPT> ./prepare_train_data.py
(venv) PROMPT> ./run_finetune.sh
main: seed: 18341
main: model base = '/Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf'
llama_model_loader: loaded meta data with 21 key-value pairs and 291 tensors from /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  10:                          general.file_type u32              = 7
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
llama_model_loader: - type q8_0:  226 tensors
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
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 8.03 B
llm_load_print_meta: model size       = 7.95 GiB (8.50 BPW) 
llm_load_print_meta: general.name     = Meta-Llama-3-8B-Instruct-imatrix
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128001 '<|end_of_text|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_tensors: ggml ctx size =    0.15 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/33 layers to GPU
llm_load_tensors:        CPU buffer size =  8137.64 MiB
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
main: tokenize training data from train_data.txt
main: sample-start: <s>
main: include-sample-start: false
tokenize_file: warning: found 6097 samples (max length 939) that exceed context length of 512. samples will be cut off.
tokenize_file: warning: found 13490 samples (min length 130) that are shorter than context length of 512.
tokenize_file: total number of samples: 19587
main: number of training tokens: 8544061
main: number of unique tokens: 38
main: train data seems to have changed. restarting shuffled epoch.
main: begin training
main: work_size = 3078520 bytes (2.9 MB)
train_opt_callback: iter=     0 sample=1/19587 sched=0.000000 loss=0.000000 |->
train_opt_callback: iter=     1 sample=9/19587 sched=0.010000 loss=15.198700 dt=00:03:08 eta=13:22:20 |->
train_opt_callback: iter=     2 sample=17/19587 sched=0.020000 loss=15.100805 dt=00:03:07 eta=13:12:00 |-->
train_opt_callback: iter=     3 sample=25/19587 sched=0.030000 loss=14.575216 dt=00:03:06 eta=13:07:24 |------->
train_opt_callback: iter=     4 sample=33/19587 sched=0.040000 loss=14.330813 dt=00:03:06 eta=13:04:56 |---------->
train_opt_callback: iter=     5 sample=41/19587 sched=0.050000 loss=13.318750 dt=00:03:07 eta=13:04:52 |-------------------->
train_opt_callback: iter=     6 sample=49/19587 sched=0.060000 loss=12.930737 dt=00:03:08 eta=13:06:10 |------------------------>
train_opt_callback: iter=     7 sample=57/19587 sched=0.070000 loss=10.687959 dt=00:03:07 eta=12:56:53 |---------------------------------------------->
train_opt_callback: iter=     8 sample=65/19587 sched=0.080000 loss=9.608527 dt=00:03:05 eta=12:46:09 |--------------------------------------------------------->
train_opt_callback: iter=     9 sample=73/19587 sched=0.090000 loss=7.040740 dt=00:03:04 eta=12:38:45 |----------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-10.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    10 sample=81/19587 sched=0.100000 loss=4.512722 dt=00:03:04 eta=12:37:40 |------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    11 sample=89/19587 sched=0.110000 loss=5.361069 dt=00:03:04 eta=12:35:19 |--------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    12 sample=97/19587 sched=0.120000 loss=4.090664 dt=00:03:04 eta=12:31:19 |---------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    13 sample=105/19587 sched=0.130000 loss=4.458398 dt=00:03:05 eta=12:32:38 |------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    14 sample=113/19587 sched=0.140000 loss=4.289121 dt=00:03:05 eta=12:29:26 |-------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    15 sample=121/19587 sched=0.150000 loss=3.725845 dt=00:03:05 eta=12:23:10 |-------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    16 sample=129/19587 sched=0.160000 loss=4.246038 dt=00:03:04 eta=12:19:30 |--------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    17 sample=137/19587 sched=0.170000 loss=4.049550 dt=00:03:04 eta=12:16:21 |---------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    18 sample=145/19587 sched=0.180000 loss=2.656627 dt=00:03:04 eta=12:13:36 |------------------------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    19 sample=153/19587 sched=0.190000 loss=3.088753 dt=00:03:04 eta=12:09:03 |-------------------------------------------------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-20.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    20 sample=161/19587 sched=0.200000 loss=2.606128 dt=00:03:04 eta=12:06:52 |------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    21 sample=169/19587 sched=0.210000 loss=2.626283 dt=00:03:04 eta=12:04:25 |------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    22 sample=177/19587 sched=0.220000 loss=2.023295 dt=00:03:05 eta=12:02:09 |------------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    23 sample=185/19587 sched=0.230000 loss=2.477558 dt=00:03:05 eta=11:59:32 |-------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    24 sample=193/19587 sched=0.240000 loss=2.415696 dt=00:03:05 eta=11:56:41 |--------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    25 sample=201/19587 sched=0.250000 loss=2.675791 dt=00:03:05 eta=11:53:59 |------------------------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    26 sample=209/19587 sched=0.260000 loss=2.308359 dt=00:03:05 eta=11:50:28 |---------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    27 sample=217/19587 sched=0.270000 loss=2.482757 dt=00:03:04 eta=11:46:02 |-------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    28 sample=225/19587 sched=0.280000 loss=2.165444 dt=00:03:04 eta=11:42:38 |----------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    29 sample=233/19587 sched=0.290000 loss=2.127503 dt=00:03:04 eta=11:39:13 |------------------------------------------------------------------------------------------------------------------------------------>
save_checkpoint_lora_file: saving to checkpoint-30.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    30 sample=241/19587 sched=0.300000 loss=2.002013 dt=00:03:05 eta=11:36:57 |------------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    31 sample=249/19587 sched=0.310000 loss=2.401982 dt=00:03:07 eta=11:43:54 |--------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    32 sample=257/19587 sched=0.320000 loss=2.166519 dt=00:03:07 eta=11:41:34 |----------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    33 sample=265/19587 sched=0.330000 loss=2.146621 dt=00:03:06 eta=11:34:48 |------------------------------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    34 sample=273/19587 sched=0.340000 loss=2.018094 dt=00:03:05 eta=11:27:44 |------------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    35 sample=281/19587 sched=0.350000 loss=2.170383 dt=00:03:05 eta=11:22:47 |----------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    36 sample=289/19587 sched=0.360000 loss=2.223642 dt=00:03:05 eta=11:19:34 |----------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    37 sample=297/19587 sched=0.370000 loss=2.139296 dt=00:03:05 eta=11:15:37 |------------------------------------------------------------------------------------------------------------------------------------>
train_opt_callback: iter=    38 sample=305/19587 sched=0.380000 loss=1.979518 dt=00:03:05 eta=11:12:37 |------------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    39 sample=313/19587 sched=0.390000 loss=2.072860 dt=00:03:04 eta=11:09:01 |------------------------------------------------------------------------------------------------------------------------------------>
save_checkpoint_lora_file: saving to checkpoint-40.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    40 sample=321/19587 sched=0.400000 loss=2.209412 dt=00:03:04 eta=11:05:09 |----------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    41 sample=329/19587 sched=0.410000 loss=2.506456 dt=00:03:05 eta=11:02:59 |-------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    42 sample=337/19587 sched=0.420000 loss=2.350895 dt=00:03:04 eta=10:59:48 |--------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    43 sample=345/19587 sched=0.430000 loss=2.009398 dt=00:03:05 eta=10:56:45 |------------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    44 sample=353/19587 sched=0.440000 loss=1.887591 dt=00:03:04 eta=10:53:18 |-------------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    45 sample=361/19587 sched=0.450000 loss=2.431976 dt=00:03:05 eta=10:50:55 |--------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    46 sample=369/19587 sched=0.460000 loss=2.153831 dt=00:03:04 eta=10:47:13 |----------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    47 sample=377/19587 sched=0.470000 loss=2.495554 dt=00:03:04 eta=10:43:43 |-------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    48 sample=385/19587 sched=0.480000 loss=2.002228 dt=00:03:05 eta=10:42:01 |------------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    49 sample=393/19587 sched=0.490000 loss=1.977686 dt=00:03:04 eta=10:38:14 |------------------------------------------------------------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-50.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    50 sample=401/19587 sched=0.500000 loss=1.907119 dt=00:03:04 eta=10:34:34 |-------------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    51 sample=409/19587 sched=0.510000 loss=1.983209 dt=00:03:04 eta=10:31:06 |------------------------------------------------------------------------------------------------------------------------------------->
train_opt_callback: iter=    52 sample=417/19587 sched=0.520000 loss=1.989653 dt=00:03:04 eta=10:27:54 |------------------------------------------------------------------------------------------------------------------------------------->
^C
```

### Merge LoRA with base model

```
(venv) PROMPT> /Users/neoneye/nobackup/git/llama.cpp/export-lora --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora lora.bin --model-out game-of-life-v5.gguf
..................................................................................................................................................
```

### Interact with the model

Copy the `game-of-life-v5.gguf` into LM-Studio's cache dir.


Prompts like this.

```
SimonsCA1
live='1' wrap_x=True dead='0' wrap_y=True alive_neighbor_count=True

# Input
0,1,0,1,1
0,0,0,1,1
1,1,1,0,0
1,1,1,1,1
0,0,0,0,0
0,1,0,0,1

# Output
```

Unfortunately it seems my finetuned model, yields the same output as the original base model.
So my finetuning has failed.
