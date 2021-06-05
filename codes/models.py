import torch
import torch.autograd as autograd
import torch.nn as nn

class BERT_ner(nn.Module):
  def __init__(self, model, num_tags):
    super(BERT_ner, self).__init__()
    self.model = model
    self.start_trainsitions = nn.Parameter(torch.empty(num_tags))
    self.end_transitinos = nn.Parameter(torch.empty(num_tags))
    self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

  def reset_params(self):
    nn.init.uniform(self.start_trainsitions, -0.1,0.1)
    nn.init.uniform(self.end_trainsitions, -0.1,0.1)
    nn.init.uniform(self.trainsitions, -0.1,0.1)

  def _compute_score(self, output, tags, masks):
    seq_len, batch_size = tags.size()

    score = self.start_trainsitions[tags[0]]
    score += output[0, torch.arange(batch_size), tags[0]]
    for i in range(1, seq_len):
        score += self.transitions[tags[i-1], tags[i]]*masks[i]
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

  def forward(self, sentnece, masks, labels):
    output = model(sentence, masks, labels = labels)
    # print(output)
    logits = output.logits
    # print(logits.size())
    score = self._compute_score(logits, tags, masks)
    norm = self._compute_normalizer(logits, masks)
    score = score-norm
    return score
