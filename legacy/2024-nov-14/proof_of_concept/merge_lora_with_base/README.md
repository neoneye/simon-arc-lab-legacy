# Merge LoRA with Base model

Unsuccessful to get it running on my M1 mac. It assumes "cuda", but I have no cuda.

I have trained a LLM and now I have a LoRA model.

-rw-r--r--  1 neoneye  staff         24 Jun  7 14:27 README.md
-rw-r--r--  1 neoneye  staff        732 Jun  7 14:27 adapter_config.json
-rw-r--r--  1 neoneye  staff  167832240 Jun  7 14:27 adapter_model.safetensors
-rw-r--r--  1 neoneye  staff         29 Jun  7 14:27 config.json
-rw-r--r--  1 neoneye  staff        464 Jun  7 14:27 special_tokens_map.json
-rw-r--r--  1 neoneye  staff    9085698 Jun  7 14:27 tokenizer.json
-rw-r--r--  1 neoneye  staff      50615 Jun  7 14:27 tokenizer_config.json
-rw-r--r--  1 neoneye  staff       5176 Jun  7 14:27 training_args.bin


How do I merge my lora model with the base model?


### Install dependencies

```
PROMPT> python3 -m venv venv
PROMPT> source venv/bin/activate
(venv) PROMPT> pip install -r requirements.txt
```

### Take snapshot of requirements.txt

```
(venv) PROMPT> pip freeze > requirements.txt
```

