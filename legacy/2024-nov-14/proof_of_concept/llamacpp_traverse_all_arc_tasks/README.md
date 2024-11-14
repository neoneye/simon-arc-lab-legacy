# Install dependencies

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

# Take snapshot of requirements.txt

```
(venv) PROMPT> pip freeze > requirements.txt
```

# Run tests

```
(venv) PROMPT> sh test.sh
```
