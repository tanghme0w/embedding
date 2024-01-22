# embedding demo

Based on Huggingface library.

## What's inside
- `metadata.jsonl`: randomly generated json file, emulates the item description dataset.
- `NaNProcessing.py`: replace "NaN" strings to real NaNs. (For preprocessing GPT-generated data, usually you can ignore this).
- `json2mmap.py`: extract item embeddings and store them in numpy memory maps.

## How to use
1. Use git lfs to clone bert-base-uncased repository (or any other models with the same interface) to the current directory.
2. Run json2mmap.
