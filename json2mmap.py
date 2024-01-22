import json

keys_to_keep = ["tag1", "tag2", "title", "description"]
item_feat = []

metadata_path = "metadata.jsonl"
output_path = "raw.jsonl"

# read json file
with open(metadata_path) as mdf:
    for line in mdf:
        # deal with NaN
        item_entry = json.loads(line.replace("NaN", '""'))
        item_entry["description"] = "" if item_entry["description"] is None else item_entry["description"]
        new_entry = dict(zip(["item_id", "text"], [item_entry['item_id'], "".join([item_entry[key] + "; " for key in keys_to_keep])]))
        item_feat.append(json.dumps(new_entry) + '\n')


with open(output_path, "w") as of:
    of.writelines(item_feat)
