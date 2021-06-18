import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset

from sklearn.model_selection import train_test_split

import os


class sentence_datasets(Dataset):
    def __init__(self, sentences, labels, tokenizer, bio2idx):
        self.sentences = sentences
        self.labels = labels
        self.bio2idx = bio2idx
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def check_word(self, word):
        return "[CLS]" == word or "[SEP]" == word  #or "##" in word

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        sentence = sentence.split()
        prediction_mask = [
            1 if not self.check_word(x) else 0 for x in sentence
        ]
        inputs = self.tokenizer.convert_tokens_to_ids(sentence)
        inputs = torch.tensor(inputs, dtype=torch.long)
        attention_mask = torch.ones(inputs.size(0), dtype=torch.long)
        label = [self.bio2idx[x] for x in label]
        label = torch.tensor(label, dtype=torch.long)
        # print(inputs.size(),mask.size(), label.size())
        prediction_mask = torch.tensor(prediction_mask, dtype=torch.uint8)
        """
        if prediction_mask[1] == 0:
            print(sentence)
            print(label)
        """
        return {
            'ids': inputs,
            'attention_mask': attention_mask,
            'tags': label,
            'prediction_mask': prediction_mask
        }


def collate_fn(batch):
    sent, label, attention_mask, prediction_mask = [], [], [], []
    for elem in batch:
        sent.append(elem['ids'])
        label.append(elem['tags'])
        attention_mask.append(elem['attention_mask'])
        prediction_mask.append(elem['prediction_mask'])
    sent = torch.nn.utils.rnn.pad_sequence(sent)
    label = torch.nn.utils.rnn.pad_sequence(label)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask)
    prediction_mask = torch.nn.utils.rnn.pad_sequence(prediction_mask)
    return sent, label, attention_mask, prediction_mask


def prepare_dataloader(tokenized_sentences,
                       labels,
                       tokenizer,
                       tag_to_idx,
                       batch_size,
                       debug=False):
    dataset = sentence_datasets(tokenized_sentences, labels, tokenizer,
                                tag_to_idx)
    if debug:
        train_index, test_index = train_test_split(range(
            int(len(dataset) * 0.001)),
                                                   test_size=0.2)
    else:
        train_index, test_index = train_test_split(range(int(len(dataset))),
                                                   test_size=0.2)
    train_dataset = Subset(dataset, train_index)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    test_dataset = Subset(dataset, test_index)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)
    return train_dataloader, test_dataloader
