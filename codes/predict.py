import os
import nltk

nltk.download('punkt')
from glob import glob
import pandas as pd
from nltk.tokenize import sent_tokenize
import numpy as np

import torch

from models import get_model

from utils import get_cfg, get_device, get_idx2tag, get_tag2idx, get_tokenizer

MAX_LENGTH = 256


def predict(model, path, tokenizer, device, idx2tag):
    json_list = glob(path + "*.json")
    for json_file in json_list:
        print(json_file[len(path):])
        df = pd.read_json(json_file)
        texts = df['text']
        for text in texts:
            sentence_list = sent_tokenize(text)
            for sentence in sentence_list:
                token = tokenizer(sentence,
                                  return_attention_mask=False,
                                  return_token_type_ids=False,
                                  return_length=True,
                                  return_tensors="pt")
                tensor = token['input_ids'][:, 1:]
                # print(tensor)
                tensor = tensor.to(device)
                with torch.no_grad():
                    label = model(tensor)
                label = np.array(label)
                label = np.squeeze(label, 1)
                # print(label.shape)
                label = [idx2tag[x] for x in label]
                if 'B' in label or 'I' in label:
                    sent = tokenizer.convert_ids_to_tokens(tensor.squeeze(0))
                    # print(sent)
                    # print(label)
                    for i in range(len(label)):
                        if label[i] == "B" or label[i] == "I":
                            print(label[i], sent[i])


if __name__ == '__main__':
    CFG = get_cfg()
    tokenizer = get_tokenizer()
    idx2tag = get_idx2tag()
    tag_to_idx = get_tag2idx()
    device = get_device()
    model, _, _ = get_model(tag_to_idx, device, CFG)
    model.load_state_dict(
        torch.load('./model/model_{}_layers.pth'.format(CFG['hidden_layers']),
                   map_location=device))
    path = '../input/test/'
    predict(model, path, tokenizer, device, idx2tag)
