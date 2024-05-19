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
```
