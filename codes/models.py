import torch
import torch.autograd as autograd
import torch.nn as nn

from tqdm import tqdm

from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AdamW

class BERT_ner(nn.Module):
    def __init__(self, model, num_tags):
        super(BERT_ner, self).__init__()
        self.model = model
        self.num_tags = num_tags
        self.start_trainsitions = nn.Parameter(torch.empty(self.num_tags))
        self.end_transitinos = nn.Parameter(torch.empty(self.num_tags))
        self.transitions = nn.Parameter(torch.empty(self.num_tags, self.num_tags))

    def reset_params(self):
        nn.init.uniform(self.start_trainsitions, -0.1,0.1)
        nn.init.uniform(self.end_trainsitions, -0.1,0.1)
        nn.init.uniform(self.trainsitions, -0.1,0.1)

    def _compute_score(self, output, tags, mask):
      seq_len, batch_size = tags.size()

      score = self.start_trainsitions[tags[0]]
      score += output[0, torch.arange(batch_size), tags[0]]
      for i in range(1, seq_len):
          # print(mask[i])
          score += self.transitions[tags[i-1], tags[i]]*mask[i]
          score += output[i, torch.arange(batch_size), tags[i]]*mask[i]

      seq_ends = mask.long().sum(dim=0) - 1
      last_tags = tags[seq_ends, torch.arange(batch_size)]
      score += self.end_transitinos[last_tags]

      return score

    def _compute_normalizer(self, output, masks):
        seq_len = output.size(0)

        score = self.start_trainsitions + output[0]
        masks = masks.byte()
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_output = output[i].unsqueeze(1)
            next_score = broadcast_output + broadcast_score + self.transitions

            next_score = torch.logsumexp(next_score, dim=1)
            # print(type(masks[i]), type(next_score), type(score))
            score = torch.where(masks[i].unsqueeze(1), next_score, score)

            score += self.end_transitinos

        return torch.logsumexp(score, dim = 1)

    def forward(self, sentence, masks, labels):
        output = self.model(sentence, masks, labels = labels)
        # print(output)
        logits = output.logits
        # print(logits.size())
        score = self._compute_score(logits, labels, masks)
        norm = self._compute_normalizer(logits, masks)
        score = score-norm
        return score.sum()/masks.float().sum()

    def decode(self, sentence, mask = None):
        output = self.model(sentence).logits
        # print(output)
        if mask is None:
            mask = output.new_ones(output.shape[:2],dtype=torch.uint8)

        return self._viterbi_decode(output,mask)

    def _viterbi_decode(self, output, mask = None):
        batch_size, seq_len = mask.size()
        output = torch.transpose(output, 0, 1)
        score = self.start_trainsitions + output[0]
        history = []
        for feat in output[1:]:
            next_score = score+self.transitions + feat
            next_score, indices = next_score.max(dim=1)
            history.append(indices)
        score += self.end_transitinos
        best_tag_list = []

        _, best_last_tag = score.max(dim=1)
        best_tags = [best_last_tag.item()]
        for hist in reversed(history):
            best_last_tag = hist[best_tags[-1]]
            # print(best_last_tag)
            best_tag_list.append(best_last_tag.item())
        best_tag_list.reverse()
        return best_tag_list


def get_model(tag_to_idx, device):
    model_name = 'bert-base-uncased'
    config = BertConfig.from_pretrained(model_name, num_labels = len(tag_to_idx))
    model = BertForTokenClassification.from_pretrained(model_name, config=config)
    for param in model.parameters():
        param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=1e-3)
    ner_model = BERT_ner(model, len(tag_to_idx)).to(device)
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
        # print(output)
        #loss = torch.mean(output)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if scheduler:
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
            #loss = torch.mean(loss)
            test_loss.append(loss.item())
    return sum(test_loss)/len(test_loss)

