import json
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm

keys_to_keep = ["tag1", "tag2", "title", "description"]
all_item_ids = []
all_item_text = []
all_item_embs = []

model_name = "bert-base-uncased-local"
metadata_path = "metadata.jsonl"
text_output_path = "raw.jsonl"
mmap_emb_path = "data.mmap"
mmap_idx_path = "idx.mmap"

# init tokenizer and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# read json file
with open(metadata_path) as mdf:
    for line in tqdm(mdf.readlines(), "processing metadata jsonl "):
        # deal with NaN
        item_entry = json.loads(line.replace("NaN", '""'))

        # register item id
        all_item_ids.append(item_entry['item_id'])

        # build combined text entry
        item_text = "".join([item_entry[key] + "; " for key in keys_to_keep])
        all_item_text.append(item_text)

        # get embedding and convert to numpy
        tokenized_text = tokenizer(item_text, return_tensors='pt', padding=True, max_length=512, truncation=True).to(device)
        text_embedding = model(**tokenized_text).pooler_output.detach().numpy()
        all_item_embs.append(dict(zip(["item_id", "embedding"], [item_entry['item_id'], text_embedding])))


# write all item text to file
with open(text_output_path, "w") as of:
    for item_id, item_text in tqdm(zip(all_item_ids, all_item_text), "create raw text file: "):
        of.writelines(json.dumps(dict(zip(["item_id", "item_text"], [item_id, item_text]))) + "\n")


# generate mmap files
all_item_ids = np.asarray(all_item_ids)
all_item_embs = np.asarray(all_item_embs)

# create index map
id_mmap = np.memmap(mmap_idx_path, mode="w+", dtype=all_item_ids.dtype, shape=(np.max(all_item_ids) + 1,))
for i, item_id in tqdm(enumerate(all_item_ids), "create index mmap "):
    id_mmap[item_id] = i
id_mmap.flush()

# create metadata map
emb_mmap = np.memmap(mmap_emb_path, mode="w+", dtype=all_item_embs.dtype, shape=all_item_embs.shape)
for i, item_emb in tqdm(enumerate(all_item_embs), "create embedding mmap "):
    emb_mmap[i] = item_emb
emb_mmap.flush()

# verify results
random_id = np.random.randint(len(all_item_ids))
gt_array = np.asarray(all_item_embs[random_id])
ret_array = emb_mmap[id_mmap[random_id]]
assert np.array_equal(gt_array, ret_array, equal_nan=False)
