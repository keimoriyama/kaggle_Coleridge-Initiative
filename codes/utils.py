import numpy as np
from tqdm import tqdm
import re
import string
import time

def get_time(start, end):
    t = end - start
    h = 0
    m = 0
    s = 0
    while t >= 3600:
      h += 1
      t -= 3600
    while t >= 60:
      m += 1
      t -= 60
    s = int(t)
    time = ""
    if h != 0:
      time += str(h) + "h "
    if m != 0:
      time += str(m) + "min "
    if s != 0:
      time += str(s) + "sec"
    return time

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
