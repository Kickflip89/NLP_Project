import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.attention import SingleAttention, Linear, EncoderAttention, SelfAttention

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class Decoder(nn.Module):
    def __init__(
                    self,
                    in_dim,
                    vocab_size,
                    out_channels=512,
                    embed_dim=240,
                    num_heads=3,
                    max_len=1056,
                    num_layers=4,
                    dropout=.1,
                    conv_GLU=True
                ):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.dropout=dropout
        self.in_dim=in_dim
        self.num_heads=num_heads
        self.vocab_size = vocab_size
        
        self.fc1 = Linear(in_dim, embed_dim, dropout=dropout, bias=True)
        
        self.conv = nn.ModuleList()
        self.self_attention=nn.ModuleList()
        self.enc_attention=nn.ModuleList()    
        for i in range(num_layers):
            self.conv.append(nn.Conv1d(embed_dim, embed_dim, 5, padding=2))
            self.enc_attention.append(EncoderAttention(embed_dim, embed_dim, embed_dim))
            self.self_attention.append(SelfAttention(embed_dim, embed_dim, num_heads))
        self.fc2 = Linear(embed_dim, out_channels, dropout=dropout, bias=True)
        self.fc3 = Linear(out_channels, vocab_size, dropout=dropout, bias=True)
        
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def forward(self, X, encoder_outputs):
        encoder_a, encoder_b = encoder_outputs
        X = self.fc1(X)
        input_embed = X
        #avg_attn = None
        for conv, attention, self_attention in zip(self.conv, self.enc_attention, self.self_attention):
            residual = X
            X = conv(X.transpose(1,2)).transpose(1,2)
            r=X
            X, _enc_att = attention(X+input_embed, encoder_a, encoder_b)
            X = X + r
            #if avg_attn == None:
                #avg_attn = _enc_att
            #else:
                #avg_attn.add_(_enc_att)
            X = self_attention(X)
            X = (X+residual) * math.sqrt(.5)
            
        X = self.fc2(X)
        X = self.dropout_layer(X)
        X = X[:,-1:,:].squeeze(1)
        return self.fc3(X)#, avg_attn

class Encoder(nn.Module):
    def __init__(self, in_dim, embed_dim=240, dropout=.1, num_layers=2, conv_GLU=True):
        super(Encoder,self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.dropout=dropout
        self.in_dim = in_dim
    
        self.fc1 = Linear(in_dim, embed_dim, dropout=dropout, bias=True)
        self.conv = nn.ModuleList()
        self.attention = nn.ModuleList()
        for i in range(num_layers):
            self.conv.append(nn.Conv1d(embed_dim, embed_dim, 5, padding=2))
            self.attention.append(SingleAttention(
                self.embed_dim, self.embed_dim, self.embed_dim,
                downsample=False, head_index=0, dropout=dropout,
               bias=True, num_heads=1, conv_GLU=conv_GLU)
            )
        self.fc2 = Linear(embed_dim, embed_dim, dropout=dropout, bias=True)
    
    def forward(self, X):
        X = self.fc1(X)
        input_embed = X
        
        for conv, attention in zip(self.conv, self.attention):
            residual = X
            X = conv(X.transpose(1,2)).transpose(1,2)
            X, _attn = attention(X,X,X)
            
            X = (X + residual) * math.sqrt(.5)
        
        X = self.fc2(X)
        
        X = GradMultiply.apply(X, 1.0 / (2.0 * self.num_layers))
        
        y = X + input_embed * math.sqrt(.5)
        
        return (X, y)

    class StoryGenerator(nn.Module):
    def __init__(self, vocab_size, w2v_dim, embed_dim, prompt_len,
                 decoder_heads, decoder_layers, encoder_layers,
                 dropout=.1, conv_GLU=True):
        super(StoryGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.w2v_dim = w2v_dim
        self.embed_dim = embed_dim
        self.prompt_len = prompt_len
        self.num_heads = decoder_heads
        self.d_layers = decoder_layers
        self.e_layers = encoder_layers
        self.conv_GLU=conv_GLU
        
        self.encoder = Encoder(w2v_dim, embed_dim, dropout=dropout,
                         num_layers=2, conv_GLU=conv_GLU)
        
        self.decoder = Decoder(w2v_dim, vocab_size, out_channels=512, embed_dim=embed_dim,
                         num_heads = self.num_heads, max_len=1056, num_layers=self.d_layers,
                         dropout=dropout, conv_GLU=True)
        
    def forward(self, prompt, prev_output):
        """
            tokens are of the form B X T X C
        """
        encoder_out = self.encoder(prompt)
        decoder_out = self.decoder(prev_output, encoder_out)
        return F.softmax(decoder_out, dim=-1)
