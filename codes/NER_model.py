import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch.utils.data.dataset import Subset

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import numpy as np
from transformers.utils.dummy_pt_objects import LineByLineWithSOPTextDataset

from models import BERT_ner
from utils import *

if 'COLAB_GPU' in set(os.environ.keys()):
  BIO_LABEL = "only_bio_labeled_dataset.csv"
else:
  BIO_LABEL = "../input/only_bio_labeled_dataset.csv"

df = pd.read_csv(BIO_LABEL)

print(df)

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tag_to_idx = { "B":1, "I": 2,"O": 3, "X": 4, "[CLS]": 5, "[SEP]": 6, "[PAD]": 0}

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)

train_data = train_data_pairs(df)

tokenized_sentences, labels = splits_sentence(train_data)

from transformers import AdamW
model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name, num_labels = len(tag_to_idx))
model = BertForTokenClassification.from_pretrained(model_name, config=config)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr = 5e-5)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: 0.95**epoch)

class sentence_datasets(Dataset):
  def __init__(self, sentences, labels, tokenizer, bio2idx):
    self.sentences = sentences
    self.labels = labels
    self.bio2idx = bio2idx
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    sentence = self.sentences[index]
    label = self.labels[index]
    inputs = tokenizer(sentence, None,
                       return_tensors='pt')
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    # print(label, tokenizer.convert_ids_to_tokens(ids.squeeze(0)))
    label = [self.bio2idx[x] for x in label]
    label = torch.tensor(label, dtype = torch.long).unsqueeze(0)
    # print(ids.size(), label.size())
    return {'ids':ids, 'mask':mask, 'tags': label}

def collate_fn(batch):
    sent, label, mask = [], [], []
    for elem in batch:
        sent.append(elem['ids'].squeeze(0))
        label.append(elem['tags'].squeeze(0))
        mask.append(elem['mask'].squeeze(0))
    sent = torch.nn.utils.rnn.pad_sequence(sent)
    label = torch.nn.utils.rnn.pad_sequence(label)
    mask = torch.nn.utils.rnn.pad_sequence(mask)
    return sent, label, mask

dataset = sentence_datasets(tokenized_sentences, labels, tokenizer, tag_to_idx)

ner_model = BERT_ner(model, len(tag_to_idx)).to(device)

train_index, test_index = train_test_split(range(int(len(dataset))), test_size = 0.3)
batch_size = 32
train_dataset = Subset(dataset, train_index)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True, collate_fn= collate_fn)
test_dataset = Subset(dataset, test_index)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False, collate_fn= collate_fn)

epochs = 1

for _ in tqdm(range(epochs)):
  train_loss, test_loss = [], []
  model.train()
  for sentence, label, mask in train_dataloader:
    # print(sentence, tags)
    model.zero_grad()
    optimizer.zero_grad()
    # sentence = tokenizer(sentence, is_split_into_words=True, padding=True)['input_ids']
    sentence = sentence.to(device)
    tags =label.to(device)
    masks = mask.to(device)
    #print(sentence.size(), tags.size(), masks.size())
    output = ner_model(sentence, masks, labels = tags)
    # print(torch.mean(output, dim = 0))
    loss = torch.mean(output)
    train_loss.append(loss.item)
    loss.backward()
    optimizer.step()
    #scheduler.step()

  total_train_loss.append(sum(train_loss)/len(train_loss))

  model.eval()
  for data in test_dataloader:
    with torch.no_grad():
      sentence = data['ids'].to(device)
      tags = data['tags'].to(device)
      masks = data['mask'].to(device)
      loss = ner_model(sentence, masks, labels = tags)['loss']
      test_loss.append(loss.item())
  total_test_loss.append(sum(test_loss)/len(test_loss))

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
  output = model(input)

idx_to_tag = {v: k for k, v in tag_to_idx.items()}

ans = torch.argmax(output['logits'], dim = 2)
predict = [idx_to_tag[x.item()] for x in ans.squeeze(0)]

