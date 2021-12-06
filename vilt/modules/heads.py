import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class LinearHead(nn.Module):
    def __init__(self, dim):
        super(LinearHead, self).__init__()
        self.classifier = nn.Linear(dim, 1)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, feats, labels):
        output = self.classifier(feats)
        loss = self.bce(output, labels.unsqueeze(1)) if labels is not None else None

        return output, loss


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MLPHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

    def forward(self, x):
        x = self.dropout(x)
        x = self.cls(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x