import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PadWithin(nn.Module):
    """
        Pads the self-attention mask back to the original
        time frame
    """
    def __init__(self, stride=2):
        super(PadWithin, self).__init__()
        self.stride = stride
        
    def forward(self, feats):
        #print(feats.size(), self.stride)
        self.w = torch.zeros(self.stride, self.stride)
        self.w[0,0] = 1
        self.w = self.w.expand(1, 1, self.stride, self.stride)
        feats = feats.unsqueeze(1)
        stride = self.stride
        res = F.conv_transpose2d(feats, self.w, stride=self.stride, groups=feats.size(1)).squeeze(1)
        #print(res.size())
        return res

class Downsample(nn.Module):
    """
    Selects every nth element, where n is the index
    Based off of Fariseq implementation
    """

    def __init__(self, index):
        super(Downsample, self).__init__()
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
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units
        Fairseq implementation
    """
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
    Author's implementation
    """
    def __init__(self, in_dim, out_dim, k=3, dropout=0.0, bias=True):
        super(GLU_conv,self).__init__()

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
    """Weight-normalized Linear layer (input: B x T x C)
        Fairseq implementation
    """
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def GatedLinear(in_features, out_features, dropout=0.0, bias=True):
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units
        Fairseq implementation"""
    return nn.Sequential(
        Linear(in_features, out_features * 4, dropout, bias),
        nn.GLU(),
        Linear(out_features * 2, out_features * 2, dropout, bias),
        nn.GLU(),
        Linear(out_features, out_features, dropout, bias),
    )
    

class SingleAttention(nn.Module):
    """
        Modified from fairseq's original code to include unique padding and convolutional GLU layers
    """
    def __init__(self, out_channels, embed_dim, head_dim, downsample=True, head_index=0, dropout=0.0,
               bias=True, num_heads=1, conv_GLU=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.head_index = head_index
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.downsample = downsample
        
        if self.downsample:
            self.ds_layer = Downsample(self.head_index)
            self.pad_layer = PadWithin(self.head_index+1)
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
        
    def MaskedSelfAttention(self, query, key, tgt_len):
        src_len = key.size()[1]
        q = query
        k = key.permute(0,2,1)
        attn_weights = torch.bmm(q, k)
        attn_weights *= torch.tril(
            attn_weights.data.new([1]).expand(src_len,src_len).clone(),
            diagonal=-1).unsqueeze(0)
        attn_weights += torch.triu(
            attn_weights.data.new([-1000]).expand(src_len,src_len).clone(),
            diagonal=0).unsqueeze(0)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.downsample and self.head_index > 0:
            attn_weights = self.pad_layer(attn_weights)
            attn_weights = attn_weights[:,:tgt_len, :tgt_len]
        return attn_weights
        
    def forward(self, k,v,q):
        batch_size, tgt_len, channels = k.size()
        """
        Scaled dot-product attention (Attention is all you need, Vaswani et. al):
        Compute bmm(Softmax(bmm(q,k^T)), v)
        """
        if self.downsample:
            k = self.ds_layer(k)
            q = self.ds_layer(q)
        q = self.queries(q)
        k = self.keys(k)
        v = self.values(v)
        q *= self.scaling
        
        #mask future timesteps
        if self.downsample:
            attn_weights = self.MaskedSelfAttention(q,k, tgt_len)
        else:
            attn_weights = torch.bmm(q,k.transpose(1,2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_weights = self.dropout(attn_weights)
        attn = torch.bmm(attn_weights, v)
        
        attn = self.out(attn)
        
        return attn, attn_weights
    
    
class MultiHeadAttention(nn.ModuleList):
    """
        Modified version of fairseq's class
    """
    def __init__(self,
                 out_channels,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 bias=True,
                 downsample=True,
                 conv_GLU=True):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.downsample = downsample
        self.conv_GLU = conv_GLU
        assert self.head_dim * num_heads == embed_dim
        
        if self.downsample:
            attention_heads = []
            for index in range(num_heads):
                attention_heads.append(
                    SingleAttention(
                        out_channels, self.embed_dim, self.head_dim,
                        self.downsample, index, dropout, bias,
                        self.num_heads, self.conv_GLU 
                    )
                )
            super().__init__(modules=attention_heads)
            self.out = Linear(embed_dim, out_channels, dropout=dropout, bias=bias)
        else:
            super().__init__()
            self.attention_module = SingleAttention(
                out_channels, self.embed_dim, self.head_dim,
                self.downsample, 1, dropout, bias,
                self.num_heads, self.conv_GLU
            )

    def forward(self,k,v,q):
        attn_list = []
        attn_weight_list = []
        if self.downsample:
            for head_index in range(self.num_heads):
                attn, attn_weight = self[head_index](k,v,q)
                attn_list.append(attn)
                attn_weight_list.append(attn_weight)
            full_attn = torch.cat(attn_list, dim=2)
            full_attn = self.out(full_attn)
            return full_attn
        else:
            attn, attn_weight = self.attention(k,v,q)
            attn_list.append(attn)
            attn_weight_list.append(attn_weight_list)
            full_attn = torch.cat(attn_list, dim=2)
            return full_attn
        
class SelfAttention(nn.Module):
    """
        wrapper class for the decoder
    """
    def __init__(self, out_channels, embed_dim, num_heads, dropout=.1, bias=True, conv_GLU=True):
        super(SelfAttention, self).__init__()
        
        self.q = Linear(out_channels, embed_dim, dropout, bias)
        self.k = Linear(out_channels, embed_dim, dropout, bias)
        self.v = Linear(out_channels, embed_dim, dropout, bias)
        
        self.attention = MultiHeadAttention(out_channels, embed_dim, num_heads, dropout, bias,
                                            downsample=True, conv_GLU=conv_GLU)
        
        self.ln = nn.LayerNorm(out_channels)
        
    def forward(self, X):
        res = X
        
        q = self.q(X)
        k = self.k(X)
        v = self.v(X)
        X = self.attention(q,k,v)
        return self.ln(X+res)
    
class EncoderAttention(nn.Module):
    """
        Unique class for single-headed encoder
    """
    def __init__(self, out_channels, embed_dim, head_dim, head_index=0, dropout=0.0,
               bias=True, conv_GLU=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.head_index = head_index
        self.head_dim = head_dim

        out_size = self.head_dim
        
        if conv_GLU:
            self.keys = GLU_conv(self.embed_dim, out_size, dropout=dropout, bias=bias)
            self.values = GLU_conv(self.embed_dim, out_size, dropout=dropout, bias=bias)
        else:
            self.keys = GatedLinear(self.embed_dim, out_size, dropout=dropout, bias=bias)
            self.values = GatedLinear(self.embed_dim, out_size, dropout=dropout, bias=bias)
            
        self.queries = GatedLinear(self.embed_dim, out_size, bias=bias)
        
        self.out = Linear(out_size, out_channels, bias=bias)
            
        self.scaling = self.head_dim ** -0.5
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value):
        batch_size, src_len, channels = key.size()
        tgt_len = query.size(1)
        """
        Scaled dot-product attention (Attention is all you need, Vaswani et. al):
        Compute bmm(Softmax(bmm(q,k^T)), v).  Here the keys and values are from
        the encoder while the query is from the decoder.
        """
        q = self.queries(query)
        k = self.keys(key).permute(0,2,1)
        v = self.values(value)
        q *= self.scaling

        attn_weights = torch.bmm(q, k)      
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn = torch.bmm(attn_weights, v)
        
        attn = self.out(attn)
        
        return attn, attn_weights
