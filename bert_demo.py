import torch
from transformers import AutoModel, AutoTokenizer

item_desc = ["newsscienceandtechnology", "jiooi iojioj hihi fewjoi", "hello!"]

model_name = "bert-base-uncased-local"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if torch.cuda.is_available():
    tokenizer.to('cuda')
tokenized_sentence = tokenizer(item_desc, return_tensors='pt', padding=True, max_length=512, truncation=True)


model = AutoModel.from_pretrained(model_name)
if torch.cuda.is_available():
    model.to('cuda')
encoded_sentence = model(**tokenized_sentence)

print(max(len(seq) for seq in tokenized_sentence['input_ids']))
print(encoded_sentence.pooler_output.shape)
