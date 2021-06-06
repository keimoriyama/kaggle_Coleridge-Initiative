import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from transformers import BertTokenizer, BertConfig, BertForTokenClassification

from tqdm import tqdm


import numpy as np

# 自作モジュール
from models import BERT_ner, get_model, train_model, val_model
from utils import train_data_pairs, splits_sentence
from data import prepare_dataloader

if 'COLAB_GPU' in set(os.environ.keys()):
  BIO_LABEL = "../../only_bio_labeled_dataset.csv"
else:
  BIO_LABEL = "../input/only_bio_labeled_dataset.csv"

df = pd.read_csv(BIO_LABEL)

print(df)

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tag_to_idx = { "B":1, "I": 2,"O": 3, "X": 4, "[CLS]": 5, "[SEP]": 6, "[PAD]": 0}

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)

train_data = train_data_pairs(df)

tokenized_sentences, labels = splits_sentence(train_data, tokenizer)

model, optimizer = get_model(tag_to_idx, device)

train_dataloader, test_dataloader = prepare_dataloader(tokenized_sentences, labels, tokenizer, tag_to_idx)

epochs = 1

for epoch in epochs:
    model, train_loss = train_model(model, optimizer, train_dataloader, device)
    val_loss = val_model(model, train_dataloader, device)
    print(f"epoch:{epoch}")
    print(f"train_loss:{train_loss}    val_loss:{val_loss}")

torch.save(model.state_dict(), './model.pth')

model.load_state_dict(torch.load('./model.pth'))

d = df.iloc[1]
sentence = d['sentence']
label = d['BIO_label']

input = tokenizer.encode(sentence)
input = torch.tensor(input, dtype = torch.long)
input = input.unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
  tags = model.decode(input)

idx_to_tag = {v: k for k, v in tag_to_idx.items()}

predict = [idx_to_tag[x.item()] for x in ans.squeeze(0)]

