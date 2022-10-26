import torch
from torch import nn
import os.path as op
import numpy as np
from tensorboardX import SummaryWriter
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F

class GRU_predictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=1, dropout_rate=0):
        super().__init__()
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first = True)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = nn.ReLU()
        self.l_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lens):
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden_t = self.gru(x)
        if self.num_layers == 1:
            hn = hidden_t.squeeze(0)
        else:
            hn = hidden_t[-1].squeeze(0)
        hn = self.dropout(hn)
        
        out = self.l_out(hn).squeeze(-1)

        return out


