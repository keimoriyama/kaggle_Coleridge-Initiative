import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertForTokenClassification
from transformers import AdamW

from torchcrf import CRF

import numpy as np


class BERT_ner(nn.Module):
    def __init__(self, model, num_tags, CRF):
        super(BERT_ner, self).__init__()
        self.model = model
        self.hidden2tags = nn.Linear(768, num_tags)
        self.CRF = CRF(num_tags)

    def forward(self, sentence, masks=None, labels=None, prediction_mask=None):
        output = self.model(sentence, masks, output_hidden_states=True)
        # print(output)
        hidden = output.logits
        # hidden = self.hidden2tags(hidden)
        if labels is not None:
            # [CLS]
            hidden = hidden[1:]
            labels = labels[1:]
            prediction_mask = prediction_mask[1:]
            print(prediction_mask[0].all())
            loss = -self.CRF(F.log_softmax(hidden, 2),
                             labels,
                             prediction_mask,
                             reduction='mean')
            return loss
        else:
            hidden = hidden[:, 1:]
            pred = self.CRF.decode(hidden)
            return pred


def get_model(tag_to_idx, device, CFG):
    model_name = 'bert-base-uncased'
    if CFG['bert_type'] == 1:
        model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=len(tag_to_idx))
    else:
        config = BertConfig.from_pretrained(
            model_name,
            num_hidden_layers=CFG['hidden_layers'],
            num_labels=len(tag_to_idx))
        model = BertForTokenClassification(config)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    ner_model = BERT_ner(model, len(tag_to_idx), CRF).to(device)
    print(ner_model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
    return ner_model, optimizer, scheduler


def train_model(model, optimizer, train_dataloader, device, scheduler=None):
    train_loss = []
    model.train()
    for sentence, label, attention_mask, prediction_mask in train_dataloader:
        optimizer.zero_grad()
        sentence = sentence.to(device)
        tags = label.to(device)
        attention_mask = attention_mask.to(device)
        prediction_mask = prediction_mask.to(device)
        loss = model(sentence, attention_mask, tags, prediction_mask)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return model, sum(train_loss) / len(train_loss)


def val_model(model, test_dataloader, device):
    model.eval()
    test_loss = []
    for sentence, label, attention_mask, prediction_mask in test_dataloader:
        with torch.no_grad():
            sentence = sentence.to(device)
            tags = label.to(device)
            attention_mask = attention_mask.to(device)
            prediction_mask = prediction_mask.to(device)
            loss = model(sentence, attention_mask, tags, prediction_mask)
            test_loss.append(loss.item())
    return sum(test_loss) / len(test_loss)


def predict_labels(model, sentence, label, idx2tag, tokenizer, device):
    sentence = sentence.split()
    sentence.insert(0, "[CLS]")
    sentence.append("[SEP]")
    # print(sentence)
    input = tokenizer.convert_tokens_to_ids(sentence)
    input = torch.tensor(input, dtype=torch.long)
    # print(input)
    model_input = input.unsqueeze(0).to(device)
    # print(model_input)
    with torch.no_grad():
        tags = model(model_input)
    print('input sentence: ', sentence)
    print("ans: ", label)
    tags = np.array(tags)
    tags = np.squeeze(tags)
    # print(tags.shape)
    predict = [idx2tag[x] for x in tags]
    print("predict: ", predict)
