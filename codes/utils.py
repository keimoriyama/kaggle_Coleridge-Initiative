import numpy as np
from tqdm import tqdm
import re
import string

def clean_text(txt):
    text = re.sub("\s+", " ", txt)
    text = "".join([k for k in text if k not in string.punctuation])
    text = re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
    return text

def train_data_pairs(df):
  train_data = []
  for i in range(len(df)):
      strings = df.iloc[i]
      sentence = strings['string']
      label = strings["label"]
      if sentence is np.nan:
          continue
      train_data.append((sentence, label))
  return train_data

def splits_sentence(train_data, tokenizer):
  tokenized_sentences,labels = [], []
  for data in tqdm(zip(train_data), total = len(train_data)):
    sentence = data[0][0]
    label = data[0][1]
    label = label.split()
    label.insert(0, "[CLS]")
    label.append("[SEP]")
    sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_sentences.append(sentence)
    labels.append(label)
  return tokenized_sentences, labels
