import os

import pandas as pd
import torch
from transformers import BertTokenizer
import time

# 自作モジュール
from models import get_model, train_model, val_model, predict_labels
from utils import train_data_pairs, splits_sentence, get_time
from data import prepare_dataloader
from predict import predict

if 'COLAB_GPU' in set(os.environ.keys()):
    BIO_LABEL = "/content/data_for_bert.csv"
elif "KAGGLE_URL_BASE" in set(os.environ.keys()):
    BIO_LABEL = "/kaggle/input/train-dataset-coleridge/data_for_bert.csv"
else:
    BIO_LABEL = "../input/data_for_bert.csv"

df = pd.read_csv(BIO_LABEL)

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tag_to_idx = {"B": 1, "I": 2, "O": 3, "[CLS]": 4, "[SEP]": 5, "[PAD]": 0}

CFG = {
    "batch_size": 32,
    "debug": False,
    'hidden_layers': 4,
    "epoch": 10,
    "bert_type": 1
}

tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                          do_lower_case=False)

train_data = train_data_pairs(df)

tokenized_sentences, labels = splits_sentence(train_data, tokenizer)

model, optimizer, scheduler = get_model(tag_to_idx, device, CFG)

train_dataloader, test_dataloader = prepare_dataloader(
    tokenized_sentences,
    labels,
    tokenizer,
    tag_to_idx,
    batch_size=CFG['batch_size'],
    debug=CFG['debug'])

d = df.iloc[5]
sample_sentence = d['string']
ans_label = d['label']
idx_to_tag = {v: k for k, v in tag_to_idx.items()}
epochs = CFG["epoch"]
for epoch in range(epochs):
    start = time.time()

    model, train_loss = train_model(model,
                                    optimizer,
                                    train_dataloader,
                                    device,
                                    scheduler=scheduler)
    val_loss = val_model(model, test_dataloader, device)
    end = time.time()
    elapsed_time = get_time(start, end)
    print(f"epoch:{epoch+1}")
    print("time: {} train_loss:{}    val_loss:{}".format(
        elapsed_time, train_loss, val_loss))
    predict_labels(model, sample_sentence, ans_label, idx_to_tag, tokenizer,
                   device)

if CFG['bert_type'] == 0:
    torch.save(
        model.state_dict(),
        './{}_model_{}_layers.pth'.format(CFG['bert_type'],
                                          CFG['hidden_layers']))
else:
    torch.save(model.state_dict(), './{}_model.pth'.format(CFG['bert_type']))
"""
model.load_state_dict(torch.load('./model.pth', map_location=device))

path = '../input/test/'
predict(model, path, tokenizer, device, idx_to_tag)
"""
