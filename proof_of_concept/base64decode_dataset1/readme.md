# Dataset: Base64 decode version1

This dataset is for improving base64 decoding capabilities.

`GPT 4o` is great at base64 decoding.

However `llama3` is terrible at base64 decoding.

The dataset looks like this:

```text
{"instruction": "Transform base64 to HEX", "input": "464pNBlIObA=", "output": "e3ae2934194839b0"}
{"instruction": "Decode Base64 to json", "input": "NQ==", "output": "[53]"}
{"instruction": "Base64 to Hexadecimal", "input": "ax0WaQ==", "output": "6b1d1669"}
{"instruction": "Change base64 to JSON", "input": "7MmBZO4=", "output": "[236,201,129,100,238]"}
```

# Generate dataset

```
PROMPT> python generate_dataset.py
```

This creates the file `base64-decode-v1.jsonl`.

