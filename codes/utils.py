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
      sentence = strings['sentence']
      label = strings["BIO_label"]
      if sentence is np.nan:
          continue
      train_data.append((sentence, label))
  return train_data

def splits_sentence(train_data, tokenizer):
  tokenized_sentences,labels = [], []
  for data in tqdm(zip(train_data), total = len(train_data)):
    sentence = data[0][0]
    label = data[0][1]
    expanded_label, words = [], []
    tokenized = tokenizer.tokenize(sentence)
    label = label.split()
    label_idx = 0
    for i, subwords in enumerate(tokenized):
      if "##" in subwords:
        words.append(subwords)
        expanded_label.append("X")
        if (i+1) < len(tokenized) and "#" not in tokenized[i+1]:
          label_idx+=1
      else:
        words.append(subwords)
        expanded_label.append(label[label_idx])
        if (i+1) < len(tokenized) and"#" not in tokenized[i+1]:
          label_idx+=1
    expanded_label.insert(0, "[CLS]")
    expanded_label.append("[SEP]")
    tokenized_sentences.append(sentence)
    labels.append(expanded_label)
  return tokenized_sentences, labels
