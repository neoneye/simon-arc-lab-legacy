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

It doesn't use my GPU. Only uses my CPUs.

Wait for it to generate a few checkpoints. After 20 minutes, press CTRL-C to stop it. Otherwise it continues forever.

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
