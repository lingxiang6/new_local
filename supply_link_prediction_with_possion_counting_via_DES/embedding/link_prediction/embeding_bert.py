import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
import numpy as np

device = 'cuda:0'


data = pd.read_csv('your/root')
tokenizer = BertTokenizer.from_pretrained('/embedding_model/root')
model = BertModel.from_pretrained('/embedding_model/root').to(device)




def encode_text(text, tokenizer, model):
    if not isinstance(text, str):
        text = str(text)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().flatten().numpy()



data['embedding_location'] = data['frist_results'].apply(lambda x: encode_text(x, tokenizer, model))
feature_embeddings = np.stack(data['embedding_location'].values)
np.save('/home/liubin/SGFormer/supply_chain/supply_link_prediction/embedding_location_bert.npy', feature_embeddings)

