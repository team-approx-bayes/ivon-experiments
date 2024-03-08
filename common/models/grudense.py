'''GRU followed by dense dataset in PyTorch.'''
import torch
import torch.nn as nn


class GRUDense(nn.Module):
    def __init__(self, vocab_size, num_classes, padding_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=padding_idx)
        self.gru = nn.GRU(
            input_size=128, hidden_size=128, num_layers=2, bias=True)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x, h0=None):
        self.gru.flatten_parameters()
        embedded = self.embedding(x)
        x, _ = self.gru(embedded, h0)
        # logits = self.linear(x[-1])
        logits = self.linear(x.mean(dim=0))
        return logits
