# Base64 decode dataset 

`GPT 4o` is great at base64 decoding.

However llama3 is terrible at base64 decoding.

The dataset looks like this:

```text
{"instruction": "Transform base64 to HEX", "input": "464pNBlIObA=", "output": "e3ae2934194839b0"}
{"instruction": "Decode Base64 to json", "input": "NQ==", "output": "[53]"}
{"instruction": "Base64 to Hexadecimal", "input": "ax0WaQ==", "output": "6b1d1669"}
{"instruction": "Change base64 to JSON", "input": "7MmBZO4=", "output": "[236,201,129,100,238]"}
```

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

### Run tests

```
(venv) PROMPT> sh test.sh
```

### Generate dataset

```
(venv) PROMPT> python generate_dataset.py
```

This creates the file `base64decode_dataset.jsonl`.

