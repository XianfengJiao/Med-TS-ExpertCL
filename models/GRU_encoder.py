
import torch
from torch import nn
import os.path as op
import numpy as np
from tensorboardX import SummaryWriter
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F

class GRU_encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first = True)

    def forward(self, x, lens):
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden_t = self.gru(x)
        if self.num_layers == 1:
            hn = hidden_t.squeeze(0)
        else:
            hn = hidden_t[-1].squeeze(0)
        return hn