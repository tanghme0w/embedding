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
text_output_path = "small_scale/raw.jsonl"
mmap_emb_path = "small_scale/data.mmap"
mmap_idx_path = "small_scale/idx.mmap"

# init tokenizer and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# read json file
with open(metadata_path) as mdf:
    for line in tqdm(mdf.readlines(), "processing metadata jsonl "):
        # deal with NaN
        item_entry = json.loads(line.replace("NaN", '""').replace("null", '""'))

        # register item id
        all_item_ids.append(item_entry['item_id'])

        # build combined text entry
        item_text = "".join([item_entry[key] + "; " for key in keys_to_keep])
        all_item_text.append(item_text)

        # get embedding and convert to numpy
        tokenized_text = tokenizer(item_text, return_tensors='pt', padding=True, max_length=512, truncation=True).to(device)
        text_embedding = model(**tokenized_text).pooler_output.cpu().detach().numpy()
        all_item_embs.append(dict(zip(["item_id", "embedding"], [item_entry['item_id'], text_embedding])))


# write all item text to file
with open(text_output_path, "w") as of:
    for item_id, item_text in tqdm(zip(all_item_ids, all_item_text), "create raw text file: "):
        of.writelines(json.dumps(dict(zip(["item_id", "item_text"], [item_id, item_text]))) + "\n")


# generate mmap files
ids = all_item_ids
ids.insert(0, 0)
item_ids_array = np.array(ids)

emb = [item['embedding'][0] for item in all_item_embs]
emb.insert(0, [0 for i in range(768)])
item_embs_array = np.array(emb)

# create index map
id_mmap = np.memmap(mmap_idx_path + f'_{np.max(item_ids_array) + 1}', mode="w+", dtype=np.int32, shape=(np.max(item_ids_array) + 1,))
for i, item_id in tqdm(enumerate(item_ids_array), "create index mmap "):
    id_mmap[item_id] = i
id_mmap.flush()

# create metadata map
emb_mmap = np.memmap(mmap_emb_path + f'_{item_embs_array.shape[0]}_{item_embs_array.shape[1]}', mode="w+", dtype=np.float32, shape=item_embs_array.shape)
for i, item_emb in tqdm(enumerate(item_embs_array), "create embedding mmap "):
    emb_mmap[i][:] = item_emb[:]
emb_mmap.flush()

# verify results
random_id = np.random.randint(1, len(all_item_ids))
for item in all_item_embs:
    if item['item_id'] == random_id:
        gt_array = np.array(item['embedding'], dtype=np.float32)[0]
        ret_array = emb_mmap[id_mmap[random_id]]
        assert np.array_equal(gt_array, ret_array, equal_nan=False)
        exit()
