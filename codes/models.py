import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from transformers import BertTokenizer, BertConfig, BertForTokenClassification, BertModel
from transformers import AdamW

from torchcrf import CRF

import numpy as np

class BERT_ner(nn.Module):
    def __init__(self, model, num_tags, CRF):
        super(BERT_ner, self).__init__()
        self.model = model
        self.hidden2tags = nn.Linear(768, num_tags)
        self.CRF = CRF(num_tags)


    def forward(self, sentence, masks = None, labels = None):
        output = self.model(sentence, masks, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        hidden = self.hidden2tags(hidden)
        if labels is not None:
            masks = masks.type(torch.uint8)
            loss = -self.CRF(F.log_softmax(hidden, 2), labels, masks, reduction='mean')
            return loss
        else:
            pred = self.CRF.decode(hidden)
            return pred

def get_model(tag_to_idx, device, CFG):
    model_name = 'bert-base-uncased'
    if CFG['bert_type'] == "default":
        model = BertForTokenClassification.from_pretrained(model_name)
    else:
        config = BertConfig.from_pretrained(model_name,
                                        num_hidden_layers=hidden_layers,
                                        num_labels = len(tag_to_idx))
        model = BertForTokenClassification(config)
    # print(model)
    #for param in model.parameters():
    #    param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=5e-5)
    ner_model=BERT_ner(model, len(tag_to_idx), CRF).to(device)
    print(ner_model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 30)
    return ner_model, optimizer, scheduler


def train_model(model, optimizer, train_dataloader, device, scheduler = None):
    train_loss = []
    model.train()
    for sentence, label, mask in train_dataloader:
        optimizer.zero_grad()
        sentence = sentence.to(device)
        tags = label.to(device)
        masks = mask.to(device)
        loss = model(sentence, masks, tags)
        train_loss.append(loss.item())
        # print(model.CRF.transitions)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return model, sum(train_loss)/len(train_loss)

def val_model(model, test_dataloader, device):
    model.eval()
    test_loss = []
    for sentence, label, mask in test_dataloader:
        with torch.no_grad():
            sentence = sentence.to(device)
            tags = label.to(device)
            masks = mask.to(device)
            loss = model(sentence, masks, tags)
            test_loss.append(loss.item())
    return sum(test_loss)/len(test_loss)

def predict_labels(model, sentence, label, idx2tag, tokenizer, device):
    input = tokenizer.encode(sentence)
    input = torch.tensor(input, dtype = torch.long)
    model_input = input.unsqueeze(0).to(device)
    with torch.no_grad():
      tags = model(model_input)
    print('input sentence: ', sentence)
    print("ans: ", label)
    tags = np.array(tags)
    tags = np.squeeze(tags)
    print(tags.shape)
    predict = [idx2tag[x] for x in tags]
    print("predict: ", predict)

