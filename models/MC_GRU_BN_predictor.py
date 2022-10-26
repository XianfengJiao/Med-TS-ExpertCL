import sys
import torch
from torch import nn
import os.path as op
import numpy as np
from tensorboardX import SummaryWriter
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F

sys.path.append("..")
from utils.model_utils import clones


class MC_GRU_BN_predictor(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=64, output_dim=1, num_layers=1, dropout_rate=0.5):
        super().__init__()
        self.GRUs = clones(nn.GRU(1, hidden_dim, num_layers=num_layers, batch_first = True), input_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.l_proj = torch.nn.Linear(hidden_dim * input_dim, hidden_dim)
        self.l_pred = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, x, lens):
        feature_dim = x.size(2)

        GRU_embeded_input = []
        for i in range(feature_dim):
            embeded_input = self.GRUs[i](pack_padded_sequence(x[:,:,i].unsqueeze(-1), lens.cpu(), batch_first=True, enforce_sorted=False))[1].squeeze(0) # b h
            GRU_embeded_input.append(embeded_input)
        GRU_embeded_input = torch.cat(GRU_embeded_input, dim=-1)
        GRU_embeded_input = self.dropout(GRU_embeded_input)

        embeded_input = self.l_proj(GRU_embeded_input)
        embeded_input = self.bn(embeded_input)
        embeded_input = self.relu(embeded_input)
        output = self.l_pred(embeded_input)
        output = self.sigmoid(output).squeeze(-1)

        return output

