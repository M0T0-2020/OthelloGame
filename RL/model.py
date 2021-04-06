import os, sys

import numpy as np
import time
import copy

from torch import nn
import torch
from torch.distributions import Categorical

class basis_conv2d(nn.Module):
    def __init__(self, input_dim):
        super(basis_conv2d, self).__init__()
        
        self.input_dim = input_dim

        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 1),
            nn.Dropout2d(0.1)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(x + self.conv_1(x))
        return x

class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super(AttentionEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()


    def forward(self,x):
        x2, attention = self.self_attn(x, x, x, attn_mask=None, key_padding_mask=None)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x, attention

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
    def forward(self, x):
        output = x
        for i, mod in enumerate(self.layers):
            output, atn = mod(output)
            if i==0:
                attention = atn
        return output, attention


class A2C_Model(nn.Module):
    def __init__(self, input_dim):
        super(A2C_Model, self).__init__()
        
        self.input_dim = input_dim
        
        self.transformer_encoder_1 = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=input_dim, nhead=input_dim),
                    num_layers=3
                )

        self.conv_1 = nn.Sequential(
            basis_conv2d(input_dim),
            nn.Conv2d(input_dim, input_dim*8, 3),
            nn.Dropout2d(0.3),
            nn.ReLU(),)

        self.conv_2 = nn.Sequential(
            basis_conv2d(input_dim*8),
            nn.Conv2d(input_dim*8, input_dim*16, 3),)
        
        self.fc_p = nn.Sequential(
            nn.Linear(768,256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64,64)
        )
        
        self.fc_v = nn.Sequential(
            nn.Linear(768,256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64,1)
        )

        self.fc_r = nn.Sequential(
            nn.Linear(768,256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    
    def attention_forward(self, x, layer):
        
        N =  x.size(0)
        inp_dim = x.size(1)
        h_dim = x.size(2)
        w_dim = x.size(3)
        att_inp = torch.flatten(x,2).permute(2,0,1)#（バッチサイズ, 次元数, W, H）=>（バッチサイズ, 次元数,系列長）=>（系列長, バッチサイズ, 次元数）
        att_inp = layer(att_inp)
        att_inp = att_inp.permute(1,2,0)
        
        att_inp = att_inp.reshape(N, inp_dim, h_dim, w_dim)
        return att_inp
    
    def forward(self, x):
        x_2 = copy.deepcopy(x[:,2,:,:].flatten(1))
        x = self.attention_forward(x, self.transformer_encoder_1)
        
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = torch.flatten(x, 1)
        
        p = self.fc_p(x)
        p[x_2==0]=0
        #p*=p_2.to(torch.int64)
        v = self.fc_v(x)
        r = self.fc_r(x)
        return {'policy':p, 'value':v, 'return':r}


class DQN_Model(nn.Module):
    def __init__(self, input_dim):
        super(DQN_Model, self).__init__()
        
        self.input_dim = input_dim
        
        self.transformer_encoder_1 = Encoder(
                    AttentionEncoderLayer(d_model=input_dim, nhead=input_dim),
                    num_layers=3
                )

        self.conv_1 = nn.Sequential(
            basis_conv2d(input_dim),
            nn.Conv2d(input_dim, input_dim*8, 3),
            nn.Dropout2d(0.3),
            nn.ReLU(),)

        self.conv_2 = nn.Sequential(
            basis_conv2d(input_dim*8),
            nn.Conv2d(input_dim*8, input_dim*16, 3),)
        
        self.fc_p = nn.Sequential(
            nn.Linear(768,768),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(768,64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64,64)
        )
        
        self.fc_v = nn.Sequential(
            nn.Linear(768,768),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(768,64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    
    def attention_forward(self, x, layer):
        
        N =  x.size(0)
        inp_dim = x.size(1)
        h_dim = x.size(2)
        w_dim = x.size(3)
        att_inp = torch.flatten(x,2).permute(2,0,1)#（バッチサイズ, 次元数, W, H）=>（バッチサイズ, 次元数,系列長）=>（系列長, バッチサイズ, 次元数）
        att_inp, attention = layer(att_inp)
        att_inp = att_inp.permute(1,2,0)
        
        att_inp = att_inp.reshape(N, inp_dim, h_dim, w_dim)
        return att_inp, attention
    
    def forward(self, x):
        x_2 = copy.deepcopy(x[:,2,:,:].flatten(1))
        x, attention = self.attention_forward(x, self.transformer_encoder_1)
        
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = torch.flatten(x, 1)
        
        q = self.fc_p(x)
        q[x_2==0]=0
        return {'policy':q, 'attention':attention}