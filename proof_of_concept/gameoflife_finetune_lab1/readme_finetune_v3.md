# Finetune v3

2 hours after I initiated the finetuning of v3, I realized I had made a grave mistake.
The way I insert junk spaces into the "input" grid. Here I also insert the column separator.
This causes the number of columns to vary.
I want the number of columns to be the same for all rows.
I want some junk spaces that messes with the alignment of the grid.

v3: I have removed the `<SFT>` separator from the prompt. I have removed the `--sample-start '<SFT>'` parameter.
v2+v1: had a `<SFT>` separator between each row.

v3: I have added a `<|eot_id|>` to the end of the prompt. So the end of the prompt looks like this `<|start_header_id|>assistant<|end_header_id|>{assistant}<|eot_id|>`. Hopefully this prevents the LLM from generating text after the result grid.
v2+v1: I had no end token. This could explain why the LLM would continue to generate text.

v3: I use the system prompt: `You are a helpful assistant.`.
v2: I used the `instruction` field as the system prompt.
v1: I used the system prompt: `Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.`, that is the same as in the original Alpaca format.

v3: Medium prompt format using convert_to_llama_format_v3.
v2: Shorter prompt format using convert_to_llama_format_v2.
v1: had longer prompt format using convert_to_llama_format_v1.

v3: Longer context length, 512. 
v2: Longer context length, 512. 
v1: had 128. Lots of warnings about most of the data being too long.

v3: I'm not using the `--adam-iter 1024` parameter.
v2: I have removed --adam-iter 1024, so now the default value is used.
v1: had --adam-iter 1024

v3: I see no info about `tokenize_file`. Is this because llama3 uses the `<|begin_of_text|>` as the separator?
It's unclear to me if it considers all the rows, or are ignoring some rows.
v2: Previously I saw these warnings, indicating that there were issues with some of the rows. Not sure how many of the rows got used for training. It seemed like the majority got rejected due to exceeding the context length.
tokenize_file: warning: found X samples (max length Y) that exceed context length of 512. samples will be cut off.
tokenize_file: warning: found Z samples (min length W) that are shorter than context length of 512.
tokenize_file: total number of samples: T


```
(venv) PROMPT> /Users/neoneye/nobackup/git/llama.cpp/finetune --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora-out lora.bin --train-data game_of_life_llama3_prompts.txt --ctx 512

main: seed: 1716028590
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
tokenize_file: total number of samples: 4089811
main: number of training tokens: 4090323
main: number of unique tokens: 1338
main: train data seems to have changed. restarting shuffled epoch.
main: begin training
main: work_size = 3078520 bytes (2.9 MB)
train_opt_callback: iter=     0 sample=1/4089811 sched=0.000000 loss=0.000000 |->
train_opt_callback: iter=     1 sample=9/4089811 sched=0.010000 loss=14.194803 dt=00:03:25 eta=14:33:04 |->
train_opt_callback: iter=     2 sample=17/4089811 sched=0.020000 loss=13.887305 dt=00:03:13 eta=13:40:19 |---->
train_opt_callback: iter=     3 sample=25/4089811 sched=0.030000 loss=14.543765 dt=00:03:11 eta=13:28:14 |>
train_opt_callback: iter=     4 sample=33/4089811 sched=0.040000 loss=14.123890 dt=00:03:09 eta=13:15:07 |-->
train_opt_callback: iter=     5 sample=41/4089811 sched=0.050000 loss=13.210417 dt=00:03:07 eta=13:05:39 |----------->
train_opt_callback: iter=     6 sample=49/4089811 sched=0.060000 loss=14.713701 dt=00:03:06 eta=12:58:34 |>
train_opt_callback: iter=     7 sample=57/4089811 sched=0.070000 loss=14.181762 dt=00:03:08 eta=13:02:51 |->
train_opt_callback: iter=     8 sample=65/4089811 sched=0.080000 loss=13.257481 dt=00:03:07 eta=12:54:45 |---------->
train_opt_callback: iter=     9 sample=73/4089811 sched=0.090000 loss=12.886841 dt=00:03:07 eta=12:51:01 |-------------->
save_checkpoint_lora_file: saving to checkpoint-10.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    10 sample=81/4089811 sched=0.100000 loss=12.613787 dt=00:03:06 eta=12:45:35 |----------------->
train_opt_callback: iter=    11 sample=89/4089811 sched=0.110000 loss=11.750807 dt=00:03:06 eta=12:41:44 |------------------------->
train_opt_callback: iter=    12 sample=97/4089811 sched=0.120000 loss=11.353323 dt=00:03:05 eta=12:33:53 |----------------------------->
train_opt_callback: iter=    13 sample=105/4089811 sched=0.130000 loss=10.574235 dt=00:03:05 eta=12:29:26 |------------------------------------->
train_opt_callback: iter=    14 sample=113/4089811 sched=0.140000 loss=10.587626 dt=00:03:05 eta=12:27:47 |------------------------------------->
train_opt_callback: iter=    15 sample=121/4089811 sched=0.150000 loss=11.282785 dt=00:03:06 eta=12:27:07 |------------------------------>
train_opt_callback: iter=    16 sample=129/4089811 sched=0.160000 loss=10.167913 dt=00:03:05 eta=12:22:15 |----------------------------------------->
train_opt_callback: iter=    17 sample=137/4089811 sched=0.170000 loss=11.745357 dt=00:03:08 eta=12:32:40 |------------------------->
train_opt_callback: iter=    18 sample=145/4089811 sched=0.180000 loss=10.742889 dt=00:03:13 eta=12:47:15 |------------------------------------>
train_opt_callback: iter=    19 sample=153/4089811 sched=0.190000 loss=12.828064 dt=00:03:14 eta=12:46:57 |--------------->
save_checkpoint_lora_file: saving to checkpoint-20.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    20 sample=161/4089811 sched=0.200000 loss=12.445628 dt=00:03:13 eta=12:41:02 |------------------>
train_opt_callback: iter=    21 sample=169/4089811 sched=0.210000 loss=8.945391 dt=00:03:10 eta=12:24:42 |----------------------------------------------------->
train_opt_callback: iter=    22 sample=177/4089811 sched=0.220000 loss=9.449986 dt=00:03:10 eta=12:22:40 |------------------------------------------------>
train_opt_callback: iter=    23 sample=185/4089811 sched=0.230000 loss=10.381705 dt=00:03:06 eta=12:04:14 |--------------------------------------->
train_opt_callback: iter=    24 sample=193/4089811 sched=0.240000 loss=10.162044 dt=00:03:05 eta=11:55:58 |----------------------------------------->
train_opt_callback: iter=    25 sample=201/4089811 sched=0.250000 loss=9.300703 dt=00:03:03 eta=11:46:36 |-------------------------------------------------->
train_opt_callback: iter=    26 sample=209/4089811 sched=0.260000 loss=9.148158 dt=00:03:05 eta=11:49:46 |--------------------------------------------------->
train_opt_callback: iter=    27 sample=217/4089811 sched=0.270000 loss=8.495910 dt=00:03:05 eta=11:49:43 |---------------------------------------------------------->
train_opt_callback: iter=    28 sample=225/4089811 sched=0.280000 loss=8.650867 dt=00:03:04 eta=11:39:15 |-------------------------------------------------------->
train_opt_callback: iter=    29 sample=233/4089811 sched=0.290000 loss=8.803698 dt=00:03:08 eta=11:51:56 |------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-30.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    30 sample=241/4089811 sched=0.300000 loss=9.086102 dt=00:03:05 eta=11:36:59 |---------------------------------------------------->
train_opt_callback: iter=    31 sample=249/4089811 sched=0.310000 loss=9.073725 dt=00:03:04 eta=11:33:23 |---------------------------------------------------->
train_opt_callback: iter=    32 sample=257/4089811 sched=0.320000 loss=8.932762 dt=00:03:07 eta=11:38:22 |------------------------------------------------------>
train_opt_callback: iter=    33 sample=265/4089811 sched=0.330000 loss=9.258301 dt=00:03:05 eta=11:30:41 |-------------------------------------------------->
train_opt_callback: iter=    34 sample=273/4089811 sched=0.340000 loss=8.862420 dt=00:03:04 eta=11:23:39 |------------------------------------------------------>
train_opt_callback: iter=    35 sample=281/4089811 sched=0.350000 loss=9.810781 dt=00:03:05 eta=11:24:53 |--------------------------------------------->
train_opt_callback: iter=    36 sample=289/4089811 sched=0.360000 loss=8.715102 dt=00:03:04 eta=11:17:47 |-------------------------------------------------------->
train_opt_callback: iter=    37 sample=297/4089811 sched=0.370000 loss=10.130274 dt=00:03:04 eta=11:12:15 |------------------------------------------>
train_opt_callback: iter=    38 sample=305/4089811 sched=0.380000 loss=9.500580 dt=00:03:03 eta=11:08:14 |------------------------------------------------>
train_opt_callback: iter=    39 sample=313/4089811 sched=0.390000 loss=8.826471 dt=00:03:04 eta=11:08:06 |------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-40.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    40 sample=321/4089811 sched=0.400000 loss=7.320676 dt=00:03:05 eta=11:06:14 |---------------------------------------------------------------------->
train_opt_callback: iter=    41 sample=329/4089811 sched=0.410000 loss=7.296052 dt=00:03:05 eta=11:03:12 |---------------------------------------------------------------------->
train_opt_callback: iter=    42 sample=337/4089811 sched=0.420000 loss=7.403055 dt=00:03:04 eta=10:59:33 |--------------------------------------------------------------------->
train_opt_callback: iter=    43 sample=345/4089811 sched=0.430000 loss=7.911721 dt=00:03:05 eta=10:58:38 |---------------------------------------------------------------->
train_opt_callback: iter=    44 sample=353/4089811 sched=0.440000 loss=7.846818 dt=00:03:05 eta=10:54:01 |---------------------------------------------------------------->
train_opt_callback: iter=    45 sample=361/4089811 sched=0.450000 loss=7.187743 dt=00:03:15 eta=11:28:42 |----------------------------------------------------------------------->
train_opt_callback: iter=    46 sample=369/4089811 sched=0.460000 loss=6.733785 dt=00:03:15 eta=11:25:58 |---------------------------------------------------------------------------->
train_opt_callback: iter=    47 sample=377/4089811 sched=0.470000 loss=7.154920 dt=00:03:13 eta=11:13:08 |----------------------------------------------------------------------->
train_opt_callback: iter=    48 sample=385/4089811 sched=0.480000 loss=6.360243 dt=00:03:06 eta=10:47:27 |------------------------------------------------------------------------------->
train_opt_callback: iter=    49 sample=393/4089811 sched=0.490000 loss=6.191371 dt=00:03:06 eta=10:44:53 |--------------------------------------------------------------------------------->
save_checkpoint_lora_file: saving to checkpoint-50.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    50 sample=401/4089811 sched=0.500000 loss=6.229803 dt=00:03:06 eta=10:40:32 |--------------------------------------------------------------------------------->
train_opt_callback: iter=    51 sample=409/4089811 sched=0.510000 loss=6.377202 dt=00:03:06 eta=10:37:30 |------------------------------------------------------------------------------->
train_opt_callback: iter=    52 sample=417/4089811 sched=0.520000 loss=6.148430 dt=00:03:06 eta=10:34:04 |--------------------------------------------------------------------------------->
train_opt_callback: iter=    53 sample=425/4089811 sched=0.530000 loss=5.477872 dt=00:03:06 eta=10:30:51 |---------------------------------------------------------------------------------------->
train_opt_callback: iter=    54 sample=433/4089811 sched=0.540000 loss=5.601840 dt=00:03:06 eta=10:28:03 |--------------------------------------------------------------------------------------->
train_opt_callback: iter=    55 sample=441/4089811 sched=0.550000 loss=6.369099 dt=00:03:06 eta=10:24:51 |------------------------------------------------------------------------------->
train_opt_callback: iter=    56 sample=449/4089811 sched=0.560000 loss=6.045095 dt=00:03:07 eta=10:23:33 |---------------------------------------------------------------------------------->
train_opt_callback: iter=    57 sample=457/4089811 sched=0.570000 loss=5.658010 dt=00:03:08 eta=10:24:10 |-------------------------------------------------------------------------------------->
train_opt_callback: iter=    58 sample=465/4089811 sched=0.580000 loss=5.464305 dt=00:03:08 eta=10:21:15 |---------------------------------------------------------------------------------------->
train_opt_callback: iter=    59 sample=473/4089811 sched=0.590000 loss=5.284284 dt=00:03:07 eta=10:15:46 |------------------------------------------------------------------------------------------>
save_checkpoint_lora_file: saving to checkpoint-60.gguf
save_checkpoint_lora_file: saving to checkpoint-LATEST.gguf
save_as_llama_lora: saving to lora.bin
save_as_llama_lora: saving to lora.bin
train_opt_callback: iter=    60 sample=481/4089811 sched=0.600000 loss=5.201558 dt=00:03:08 eta=10:15:34 |------------------------------------------------------------------------------------------->
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
(venv) PROMPT> /Users/neoneye/nobackup/git/llama.cpp/export-lora --model-base /Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q6_K.gguf --lora lora.bin --model-out game-of-life-v3.gguf
..................................................................................................................................................
```

### Interact with the model

Copy the `game-of-life-v3.gguf` into LM-Studio's cache dir.

Launch LM-Studio.

Pick the model. 

Set `temperature=0`.

Set the system prompt to the following:
```text
You are a helpful assistant.
```

Ask the model the following:

```text
Game of Life. alive='β' wrap=none dead='α'

βαααβ
ααααα
αβββα
ααααα
βαααβ
```

Answer from the model is the following:

```text
A Game of Life!

Let's simulate the next generation:

1. βαααβ:
	* β (alive) has 3 alive neighbors, stays alive.
	* α (dead) has 2 alive neighbors, becomes alive.
	* α (dead) has 2 alive neighbors, becomes alive.
	* α (dead) has 2 alive neighbors, becomes alive.
	* α (dead) has 1 alive neighbor, stays dead.
	* β (alive) has 3 alive neighbors, stays alive.

Result: βαααβααα

2. ααααα:
	* α (alive) has 4 alive neighbors, stays alive.

Result: ααααα

3. αβββα:
	* α (dead) has 1 alive neighbor, stays dead.
	* β (alive) has 2 alive
```

The response is not resembling what I had in mind.
I had expected to see a grid, without all the reasoning steps being explained.


Ask the model the following:

```text
Game of Life. alive='█' wrap=xy dead='💀'

█💀💀💀█
💀💀💀💀💀
💀███💀
💀💀💀💀💀
█💀💀💀█
```

Answer from the model is the following:

```text
A Game of Life simulation!

Here's the initial state:
long winded explanation.
```

Ask the model the following:

```text
Game of Life. live='█' wrapx=true wrapy=true dead='💀'

# Input:
█💀💀💀█
💀💀💀💀💀
💀███💀
💀💀💀💀💀
█💀💀💀█

# Output:
```

Ask the model the following:

```text
Game of Life. live='█' wrapx wrapy dead='💀'

# Input
█💀💀💀█
💀💀💀💀💀
💀███💀
💀💀💀💀💀
█💀💀💀█

# Output
```

Ask the model the following:

```text
Game of Life. live='█' wrap=both dead='💀'

# Input
█💀💀💀█
💀💀💀💀💀
💀███💀
💀💀💀💀💀
█💀💀💀█

# Output
```

Ask the model the following:

```text
Game of Life. alive='1' wrap=xy dead='0'

1,0,0,0,1
0,0,0,0,0
0,1,1,1,0
0,0,0,0,0
1,0,0,0,1
```

Ask the model the following:

```text
Game of Life. alive='1' wrap=xy dead='0'

# Input
1,0,0,0,1
0,0,0,0,0
0,1,1,1,0
0,0,0,0,0
1,0,0,0,1

# Task A: count alive neighbours

Consider wrap around x-axis and y-axis.

# Task B: compute output
```
