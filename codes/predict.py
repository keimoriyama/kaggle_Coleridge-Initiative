import os
from glob import glob
import pandas as pd
from nltk.tokenize import sent_tokenize
import numpy as np

import torch

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
                length = token['length']
                tensor = token['input_ids']
                tensor = tensor.to(device)
                with torch.no_grad():
                    label = model(tensor)
                label = np.array(label)
                label = np.squeeze(label)
                # print(label)
                label = [idx2tag[x] for x in label]
                if 'B' in label or 'I' in label:
                    sent = tokenizer.convert_ids_to_tokens(tensor.squeeze(0))
                    for i in range(len(label)):
                        if label[i] == "B" or label[i] == "I":
                            print(label[i], sent[i])