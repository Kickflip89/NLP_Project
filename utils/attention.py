import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample(nn.Module):
    """
    Selects every nth element, where n is the index
    """

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[:, :: self.index + 1, :]

def Linear(in_features, out_features, dropout=0.0, bias=True):
    """Weight-normalized Linear layer (input: B x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def GatedLinear(in_features, out_features, dropout=0.0, bias=True):
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units"""
    return nn.Sequential(
        Linear(in_features, out_features * 4, dropout, bias),
        nn.GLU(),
        Linear(out_features * 2, out_features * 2, dropout, bias),
        nn.GLU(),
        Linear(out_features, out_features, dropout, bias),
    )

    
class GLU_conv(nn.Module):
    """
    Performs 2x convlutions with GLU activations and a linear output
    Input shape: (bs, seq_len, channels)
    Intermediate representation: (bs, channels, seq_len)
    Output shape: (bs, seq_len, channels)
    """
    def __init__(self, in_dim, out_dim, k=3, dropout=0.0, bias=True):
        super().__init__()

        #for reshaping residual if necessary:
        self.convres1 = nn.utils.weight_norm(nn.Conv1d(in_dim, out_dim*2,
                                kernel_size=1),name='weight',dim=0)
        self.convres2 = nn.utils.weight_norm(nn.Conv1d(in_dim, out_dim,
                                kernel_size=1),name='weight',dim=0)

        #left padding to prevent future timesteps at current hidden state
        self.leftpad = nn.ConstantPad1d((k-1,0), 0)

        #shape (bs, in_dim, seq_len+(k-1))
        self.conv1a = nn.utils.weight_norm(nn.Conv1d(in_dim, out_dim*2,
                        kernel_size=1),name='weight', dim=0)
        self.conv1b = nn.utils.weight_norm(nn.Conv1d(in_dim, out_dim*2,
                        kernel_size=1),name='weight', dim=0)

        #shape (bs, out_dim*2, seq_len+(k-1))
        self.conv2a=nn.utils.weight_norm(nn.Conv1d(out_dim*2,out_dim*2,
                        kernel_size=k),name='weight',dim=0)
        self.conv2b=nn.utils.weight_norm(nn.Conv1d(out_dim*2,out_dim*2,
                        kernel_size=k),name='weight',dim=0)

        #shape (bs, out_dim*2, seq_len + k-1)
        self.conv3a=nn.utils.weight_norm(nn.Conv1d(out_dim*2,out_dim,
                        kernel_size=1),name='weight',dim=0)
        self.conv3b=nn.utils.weight_norm(nn.Conv1d(out_dim*2,out_dim,
                        kernel_size=1),name='weight',dim=0)
        
        #shape (bs, out_dim*2, seq_len + k-1) 
        self.conv4a=nn.utils.weight_norm(nn.Conv1d(out_dim,out_dim,
                        kernel_size=k),name='weight',dim=0)
        self.conv4b=nn.utils.weight_norm(nn.Conv1d(out_dim,out_dim,
                        kernel_size=k),name='weight',dim=0)

        #shape (bs, seq_len, out_dim)
        self.linear = Linear(out_dim, out_dim, dropout=dropout, bias=bias)
        #out shape (bs, out_dim, seq_len)

    def forward(self, X):
        X = X.permute(0,2,1)
        res1 = self.convres1(X)
        res2 = self.convres2(X)
        X=self.leftpad(X)
        #conv1 with GLU
        Xa = self.conv2a(self.conv1a(X))
        Xb = self.conv2b(self.conv1b(X))
        Xb = torch.sigmoid(Xb)
        X = torch.mul(Xa,Xb)
        X = X + res1
        X = self.leftpad(X)
        #conv2 with GLU
        Xa = self.conv4a(self.conv3a(X))
        Xb = self.conv4b(self.conv3b(X))
        Xb = torch.sigmoid(Xb)
        X = torch.mul(Xa,Xb)
        X = X + res2
        X = X.permute(0,2,1)
        return self.linear(X)

def Linear(in_features, out_features, dropout=0.0, bias=True):
    """Weight-normalized Linear layer (input: B x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def GatedLinear(in_features, out_features, dropout=0.0, bias=True):
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units"""
    return nn.Sequential(
        Linear(in_features, out_features * 4, dropout, bias),
        nn.GLU(),
        Linear(out_features * 2, out_features * 2, dropout, bias),
        nn.GLU(),
        Linear(out_features, out_features, dropout, bias),
    )

def MaskedSelfAttention(query, key, value):
    src_len = key.size()[1]
    q = query
    k = key.permute(0,2,1)
    v = value
    attn_weights = torch.bmm(q, k)
    attn_weights *= torch.tril(
        attn_weights.data.new([1]).expand(src_len,src_len).clone(),
        diagonal=-1).unsqueeze(0)
    attn_weights += torch.triu(
        attn_weights.data.new([-math.inf]).expand(src_len,src_len).clone(),
        diagonal=0,
    ).unsqueeze(0)
    
    return attn_weights
    


class SingleAttention(nn.Module):
    
    def __init__(self, out_channels, embed_dim, head_dim, downsample=True, head_index=0, dropout=0.0,
               bias=True, num_heads=1, conv_GLU=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.head_index = head_index
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.projection = None
        self.downsample = downsample
        
        if self.downsample:
            self.ds_layer = Downsample(self.head_index)
            out_size = self.head_dim
        else:
            out_size = self.head_dim * self.num_heads
        
        if conv_GLU:
            self.keys = GLU_conv(self.embed_dim, out_size, dropout=dropout, bias=bias)
            self.values = GLU_conv(self.embed_dim, out_size, dropout=dropout, bias=bias)
        else:
            self.keys = GatedLinear(self.embed_dim, out_size, dropout=dropout, bias=bias)
            self.values = GatedLinear(self.embed_dim, out_size, dropout=dropout, bias=bias)
            
        self.queries = GatedLinear(self.embed_dim, out_size, bias=bias)
        
        if self.downsample:
            self.out = Linear(out_size, self.head_dim, bias=bias)
        else:
            self.out = Linear(out_size, out_channels, bias=bias)
            
        self.scaling = self.head_dim ** -0.5
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, X):
        batch_size, src_len, channels = X.size()
        
        """
        Scaled dot-product attention (Attention is all you need, Vaswani et. al):
        Compute bmm(Softmax(bmm(q,k^T)), v)
        """
        if self.downsample:
            k = self.ds_layer(X)
            v = self.ds_layer(X)
            q = self.ds_layer(X)
        else:
            k = X
            v = X
            q = X
        q = self.queries(q)
        k = self.keys(k)
        v = self.values(v)
        q *= self.scaling
        print(q.size())
        print(k.size())
        print(v.size())

        #mask future timesteps
        attn_weights = MaskedSelfAttention(q,k,v)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn = torch.bmm(attn_weights, v)
        attn = self.out(attn)
        
        return attn, attn_weights
