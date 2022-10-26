

import copy

import numpy as np
import argparse
import os
import imp
import re
import pickle
import datetime
import random
import math
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import kneighbors_graph
import torch.distributions.bernoulli as bernoulli

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, device='cuda', dim=None):
        super(Sparsemax, self).__init__()
        self.device = device
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))
        
        dim = 1
        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits+1, device=self.device, dtype=torch.float32).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

class SingleAttention(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', demographic_dim=12, time_aware=False, use_demographic=False):
        super(SingleAttention, self).__init__()
        
        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.use_demographic = use_demographic
        self.demographic_dim = demographic_dim
        self.time_aware = time_aware

        # batch_time = torch.arange(0, batch_mask.size()[1], dtype=torch.float32).reshape(1, batch_mask.size()[1], 1)
        # batch_time = batch_time.repeat(batch_mask.size()[0], 1, 1)
        
        if attention_type == 'add':
            if self.time_aware == True:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wd = nn.Parameter(torch.randn(demographic_dim, attention_hidden_dim))
            self.bh = nn.Parameter(torch.zeros(attention_hidden_dim,))
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1,))
            
            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'mul':
            self.Wa = nn.Parameter(torch.randn(attention_input_dim, attention_input_dim))
            self.ba = nn.Parameter(torch.zeros(1,))
            
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'concat':
            if self.time_aware == True:
                self.Wh = nn.Parameter(torch.randn(2*attention_input_dim+1, attention_hidden_dim))
            else:
                self.Wh = nn.Parameter(torch.randn(2*attention_input_dim, attention_hidden_dim))

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1,))
            
            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        else:
            raise RuntimeError('Wrong attention type.')
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
    
    def forward(self, input, demo=None):
 
        batch_size, time_step, input_dim = input.size() # batch_size * time_step * hidden_dim(i)
        #assert(input_dim == self.input_dim)

        # time_decays = torch.zeros((time_step,time_step)).to(device)# t*t
        # for this_time in range(time_step):
        #     for pre_time in range(time_step):
        #         if pre_time > this_time:
        #             break
        #         time_decays[this_time][pre_time] = torch.tensor(this_time - pre_time, dtype=torch.float32).to(device)
        # b_time_decays = tile(time_decays, 0, batch_size).view(batch_size,time_step,time_step).unsqueeze(-1).to(device)# b t t 1

        time_decays = torch.tensor(range(time_step-1,-1,-1), dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)# 1*t*1
        b_time_decays = time_decays.repeat(batch_size,1,1)# b t 1
        
        if self.attention_type == 'add': #B*T*I  @ H*I
            q = torch.matmul(input[:,-1,:], self.Wt)# b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim)) #B*1*H
            if self.time_aware == True:
                # k_input = torch.cat((input, time), dim=-1)
                k = torch.matmul(input, self.Wx)#b t h
                # k = torch.reshape(k, (batch_size, 1, time_step, self.attention_hidden_dim)) #B*1*T*H
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)#  b t h
            else:
                k = torch.matmul(input, self.Wx)# b t h
                # k = torch.reshape(k, (batch_size, 1, time_step, self.attention_hidden_dim)) #B*1*T*H
            if self.use_demographic == True:
                d = torch.matmul(demo, self.Wd) #B*H
                d = torch.reshape(d, (batch_size, 1, self.attention_hidden_dim)) # b 1 h
            h = q + k + self.bh # b t h
            if self.time_aware == True:
                h += time_hidden
            h = self.tanh(h) #B*T*H
            e = torch.matmul(h, self.Wa) + self.ba #B*T*1
            e = torch.reshape(e, (batch_size, time_step))# b t
        elif self.attention_type == 'mul':
            e = torch.matmul(input[:,-1,:], self.Wa)#b i
            e = torch.matmul(e.unsqueeze(1), input.permute(0,2,1)).squeeze() + self.ba #b t
        elif self.attention_type == 'concat':
            q = input[:,-1,:].unsqueeze(1).repeat(1,time_step,1)# b t i
            k = input
            c = torch.cat((q, k), dim=-1) #B*T*2I
            if self.time_aware == True:
                c = torch.cat((c, b_time_decays), dim=-1) #B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba #B*T*1
            e = torch.reshape(e, (batch_size, time_step)) # b t 
        
        # e = torch.exp(e - torch.max(e, dim=-1, keepdim=True).values)
        
        # if self.attention_width is not None:
        #     if self.history_only:
        #         lower = torch.arange(0, time_step).to(device) - (self.attention_width - 1)
        #     else:
        #         lower = torch.arange(0, time_step).to(device) - self.attention_width // 2
        #     lower = lower.unsqueeze(-1)
        #     upper = lower + self.attention_width
        #     indices = torch.arange(0, time_step).unsqueeze(0).to(device)
        #     e = e * (lower <= indices).float() * (indices < upper).float()
        
        # s = torch.sum(e, dim=-1, keepdim=True)
        # mask = subsequent_mask(time_step).to(device) # 1 t t 下三角
        # scores = e.masked_fill(mask == 0, -1e9)# b t t 下三角
        a = self.softmax(e) #B*T
        v = torch.matmul(a.unsqueeze(1), input).squeeze() #B*I

        return v, a

class FinalAttentionQKV(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, input_dim, attention_type='add', dropout=None, attention_act='softmax'):
        super(FinalAttentionQKV, self).__init__()
        
        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.attention_act = attention_act
        self.input_dim = input_dim


        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(torch.zeros(1,))
        self.b_out = nn.Parameter(torch.zeros(1,))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(torch.randn(2*attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1,))
        self.rate = nn.Parameter(torch.ones(self.input_dim))
        
        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sparsemax = Sparsemax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, mask=None, src_key_padding_mask=None):
        batch_size, time_step, input_dim = input.size() # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(torch.mean(input, dim=1)) # b h
        input_k = self.W_k(input)# b t h
        input_v = self.W_v(input)# b t h

        if self.attention_type == 'add': #B*T*I  @ H*I

            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim)) #B*1*H
            h = q + input_k + self.b_in # b t h
            h = self.tanh(h) #B*T*H
            e = self.W_out(h) # b t 1
            e = torch.reshape(e, (batch_size, time_step))# b t

        elif self.attention_type == 'mul':
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1)) #B*h 1
            e = torch.matmul(input_k, q).squeeze(-1)#b t
            if mask != None:
                time_miss = torch.log(1 + (1 - self.sigmoid(e)) * (mask.squeeze()))
                e = e - self.rate * time_miss

            
        elif self.attention_type == 'concat':
            q = input_q.unsqueeze(1).repeat(1,time_step,1)# b t h
            k = input_k
            c = torch.cat((q, k), dim=-1) #B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba #B*T*1
            e = torch.reshape(e, (batch_size, time_step)) # b t 
        
        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        
        if src_key_padding_mask is not None:
            e = e.masked_fill(src_key_padding_mask, -1e9)
        
        if self.attention_act == 'sparsemax':
            a = self.sparsemax(e) #B*T
        else:
            a = self.softmax(e) #B*T
        
        
        
        
        
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze(1) #B*I

        return v, a

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index).to(device)

class PositionwiseFeedForward(nn.Module): # new added
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))), None

class PositionalEncoding(nn.Module): # new added / not use anymore
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.5, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 # 下三角矩阵

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)# b h t d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) # b h t t
    if mask is not None:# 1 1 t t
        scores = scores.masked_fill(mask == 0, -1e9)# b h t t 下三角
    p_attn = F.softmax(scores, dim = -1)# b h t t
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # b h t v (d_k) 
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, self.d_k * self.h), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # 1 1 t t

        nbatches = query.size(0)# b
        input_dim = query.size(1)# i+1
        feature_dim = query.size(-1)# i+1

        #input size -> # batch_size * d_input * hidden_dim
        
        # d_model => h * d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] # b num_head d_input d_k
        
       
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)# b num_head d_input d_v (d_k) 
      
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)# batch_size * d_input * hidden_dim

        #DeCov 
#         DeCov_contexts = x.transpose(0, 1).transpose(1, 2) # I+1 H B
# #         print(DeCov_contexts.shape)
#         Covs = cov(DeCov_contexts[0,:,:])
#         DeCov_loss = 0.5 * (torch.norm(Covs, p = 'fro')**2 - torch.norm(torch.diag(Covs))**2 ) 
#         for i in range(11 -1):
#             Covs = cov(DeCov_contexts[i+1,:,:])
#             DeCov_loss += 0.5 * (torch.norm(Covs, p = 'fro')**2 - torch.norm(torch.diag(Covs))**2 ) 
        DeCov_loss = 0

        return self.final_linear(x), DeCov_loss

    
def lNorm(x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    eps = 1e-7
    return (x - mean) / (std + eps)

class MultiHeadedAttentionBoost(nn.Module):
    def __init__(self, h, d_model, attn_1 = 0.4, attn_2 = 50, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionBoost, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, self.d_k * self.h), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.mask = None
        self.attn_1 = attn_1
        self.attn_2 = attn_2
        
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # 1 1 t t

        nbatches = query.size(0)# b
        input_dim = query.size(1)# i+1
        feature_dim = query.size(-1)# i+1

        #input size -> # batch_size * d_input * hidden_dim
        
        # d_model => h * d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] # b num_head d_input d_k
        
        
        querys = []
        keys = []
        values = []
        for i in range(self.h):
            querys.append(query[:,i].unsqueeze(1))
            keys.append(key[:,i].unsqueeze(1))
            values.append(value[:,i].unsqueeze(1))
            
        
       
        if self.training == True:

            x, attn = attention(querys[0], keys[0], values[0], mask=mask, 
                                     dropout=self.dropout)# b num_head d_input d_v (d_k) 


            self.attn = attn
    #         self.mask = mask
            attn1 = torch.mean(attn[:,0],1)
    #         print(attn1.shape) # 256, 153
    #         attn1_p = attn1/torch.sum(attn1,1).unsqueeze(1)

            attn1_p = self.attn_1 *self.sigmoid(lNorm(attn1)*self.attn_2)
    #         print(attn1_p.shape)


            dis_b1 = bernoulli.Bernoulli(attn1_p)
            to_mask1 = 1 - dis_b1.sample()
            mask1 = to_mask1.unsqueeze(1).repeat(1,input_dim,1).unsqueeze(1)
    #             print(attn1_p)
    #             print(mask1)

    #         mask1 = torch.ones((nbatches,input_dim,2*input_dim-1)).to(device)
    #         mask1[0,0,attn1:attn1+1,:] = 0
    #         mask1[0,0,:,attn1:attn1+1] = 0

            tempx, attn = attention(querys[1], keys[1], values[1], mask=mask1, 
                                     dropout=self.dropout)
            self.attn = torch.cat((self.attn, attn), 1)
            self.mask = mask1
            x = torch.cat((x, tempx),1)

            attn2 = attn1 + torch.mean(attn[:,0],1)
    #         attn2_p = attn2/torch.sum(attn2,1).unsqueeze(1)
            attn2_p = self.attn_1 * self.sigmoid(lNorm(attn2)*self.attn_2)

            dis_b2 = bernoulli.Bernoulli(attn2_p)
            to_mask2 = 1 - dis_b2.sample()
            mask2 = to_mask2.unsqueeze(1).repeat(1,input_dim,1).unsqueeze(1)

            tempx, attn = attention(querys[2], keys[2], values[2], mask=mask2, 
                                     dropout=self.dropout)
            self.attn = torch.cat((self.attn, attn), 1)
            self.mask = torch.cat((self.mask, mask2), 1)
            x = torch.cat((x, tempx),1)

    #         attn3 = torch.mean(attn[:,0],0)
            attn3 = attn2 + torch.mean(attn[:,0],1)
    #         attn3_p = attn3/torch.sum(attn3,1).unsqueeze(1)
            attn3_p = self.attn_1 * self.sigmoid(lNorm(attn3)*self.attn_2)

            dis_b3 = bernoulli.Bernoulli(attn3_p)
            to_mask3 = 1 - dis_b3.sample()
            mask3 = to_mask3.unsqueeze(1).repeat(1,input_dim,1).unsqueeze(1)


            tempx, attn = attention(querys[3], keys[3], values[3], mask=mask3, 
                                     dropout=self.dropout)
            self.attn = torch.cat((self.attn, attn), 1)
            self.mask = torch.cat((self.mask, mask3), 1)

    #         attn4 = torch.mean(attn[0,0],0)
            x = torch.cat((x, tempx),1)
    
        

        if self.training == False:
# #             print(1)
            
            x, attn = attention(querys[0], keys[0], values[0], mask=None, 
                                     dropout=self.dropout)# b num_head d_input d_v (d_k) 


            self.attn = attn

            tempx, attn = attention(querys[1], keys[1], values[1], mask=None, 
                                     dropout=self.dropout)
            self.attn = torch.cat((self.attn, attn), 1)
            x = torch.cat((x, tempx),1)



            tempx, attn = attention(querys[2], keys[2], values[2], mask=None, 
                                     dropout=self.dropout)
            self.attn = torch.cat((self.attn, attn), 1)
            x = torch.cat((x, tempx),1)




            tempx, attn = attention(querys[3], keys[3], values[3], mask=None, 
                                     dropout=self.dropout)
            self.attn = torch.cat((self.attn, attn), 1)

    #         attn4 = torch.mean(attn[0,0],0)
            x = torch.cat((x, tempx),1)
        
#         kl = nn.functional.kl_div(self.softmax(attn3).log(), self.softmax(attn4), size_average=None, reduce=None, reduction='mean') + \
#             nn.functional.kl_div(self.softmax(attn4).log(), self.softmax(attn3), size_average=None, reduce=None, reduction='mean')
#         torch.nn.functional.kl_div(attn3, attn4, reduction='batchmean') + torch.nn.functional.kl_div(attn4, attn3, reduction='batchmean')

        #Str DeCov
        # DeCov_contexts = x.transpose(1, 2).transpose(0, 1).transpose(1, 2).transpose(2, 3)#d_input num_head d_v b (d_k) 
        # DeCov_between_loss = torch.tensor(0.0, dtype = torch.float32, device = device)
        # DeCov_inside_loss = torch.tensor(0.0, dtype = torch.float32, device = device)
        # for input_idx in range(input_dim):
        #     current_input = DeCov_contexts[input_idx,:,:,:]
        #     for i in range(self.h):
        #         for j in range(self.h):
        #             if i == j:
        #                 covs = cov(current_input[i,:,:])
        #                 DeCov_inside_loss += 0.5 * (torch.norm(covs, p = 'fro')**2)  
        #                 #print(DeCov_inside_loss)
        #             else:
        #                 Vstack_context = torch.cat((current_input[i,:,:],current_input[j,:,:]) ,0)# 2*d_v b
        #                 assert(Vstack_context.size(0) == 2*self.d_k)
        #                 assert(Vstack_context.size(1) == nbatches)
        #                 overflow_covs = cov(Vstack_context)[self.d_k:,:]
        #                 covs = overflow_covs[:,:self.d_k]
        #                 DeCov_between_loss += 0.5 * (torch.norm(covs, p = 'fro')**2) 
        #                 #print(DeCov_between_loss)
        # DeCov_loss = DeCov_between_loss + 0.1 * DeCov_inside_loss
        #print(DeCov_loss)


      
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)# batch_size * d_input * hidden_dim

        #DeCov 
#         DeCov_contexts = x.transpose(0, 1).transpose(1, 2) # I+1 H B
# #         print(DeCov_contexts.shape)
#         Covs = cov(DeCov_contexts[0,:,:])
#         DeCov_loss = 0.5 * (torch.norm(Covs, p = 'fro')**2 - torch.norm(torch.diag(Covs))**2 ) 
#         for i in range(feature_dim -1+1):
#             Covs = cov(DeCov_contexts[i+1,:,:])
#             DeCov_loss += 0.5 * (torch.norm(Covs, p = 'fro')**2 - torch.norm(torch.diag(Covs))**2 ) 

#         print(DeCov_loss)
#         print(kl)
        
        return self.final_linear(x), torch.zeros(1)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]) , returned_value[1]

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])