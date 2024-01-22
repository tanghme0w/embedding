import torch
from transformers import AutoModel, AutoTokenizer

item_desc = ["andtechnology", "newsscienceandtechnology", "news science and technology"]

model_name = "bert-base-uncased-local"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_sentence = tokenizer(item_desc, return_tensors='pt', padding=True, max_length=512, truncation=True)

# model = AutoModel.from_pretrained(model_name)
# if torch.cuda.is_available():
#     tokenized_sentence.to('cuda')
# encoded_sentence = model(**tokenized_sentence)

print(tokenized_sentence['input_ids'])