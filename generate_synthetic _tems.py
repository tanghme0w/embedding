import json
import random

# Function to generate a random string
def random_string(length=10):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(letters) for i in range(length))

# Generate 100 entries
entries = []
for i in range(100):
    entry = {
        "item_id": i,
        "tag1": random_string(),
        "tag2": random_string(),
        "title": random_string(15),
        "description": random_string(20) if random.random() > 0.1 else "NaN"  # 10% chance of NaN
    }
    entries.append(json.dumps(entry))

# Save to a JSONL file
file_path = 'metadata.jsonl'
with open(file_path, 'w') as file:
    for entry in entries:
        file.write(entry + '\n')
