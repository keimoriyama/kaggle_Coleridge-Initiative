import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset

from sklearn.model_selection import train_test_split

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
        inputs = self.tokenizer(sentence, None,
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

def prepare_dataloader(tokenized_sentences, labels, tokenizer, tag_to_idx):
    dataset = sentence_datasets(tokenized_sentences, labels, tokenizer, tag_to_idx)
    train_index, test_index = train_test_split(range(int(len(dataset))), test_size = 0.3)
    batch_size = 16
    train_dataset = Subset(dataset, train_index)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True, collate_fn= collate_fn)
    test_dataset = Subset(dataset, test_index)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False, collate_fn= collate_fn)
    return train_dataloader, test_dataloader
