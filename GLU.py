import torch
from torch import nn

class GLU_block(nn.Module):
    def __init__(self, k, in_dim, out_dim, downsamp):
        self.reshape_residual = (in_dim != out_dim)
        int_dim = in_dim//downsamp

        #for reshaping residual if necessary:
        self.convres = nn.utils.weight_norm(nn.Conv2d(in_dim, out_dim,
                                kernel_size=(1,1)),name='weight',dim=0)

        #left padding to prevent future timesteps at current hidden state
        self.leftpad = nn.ConstantPad2d((0,0,k-1,0),0)

        #shape (bs, in_dim, seq_len+(k-1),1)
        self.conv1a = nn.utils.weight_norm(nn.Conv2d(in_dim, int_dim,
                        kernel_size=(1,1)),name='weight', dim=0)
        self.conv1b = nn.utils.weight_norm(nn.Conv2d(in_dim, int_dim,
                        kernel_size=(1,1)),name='weight', dim=0)

        #shape (bs, in_dim//downsamp, seq_len+(k-1),1)
        self.conv2a=nn.utils.weight_norm(nn.Conv2d(int_dim,int_dim,
                        kernel_size=(k,1)),name='weight',dim=0)
        self.conv2b=nn.utils.weight_norm(nn.Conv2d(int_dim,int_dim,
                        kernel_size=(k,1)),name='weight',dim=0)

        #shape (bs, in_dim//downsamp, seq_len, 1)
        self.conv3a=nn.utils.weight_norm(nn.Conv2d(int_dim,out_dim,
                        kernel_size=(1,1)),name='weight',dim=0)
        self.conv3b=nn.utils.weight_norm(nn.Conv2d(int_dim,out_dim,
                        kernel_size=(1,1)),name='weight',dim=0)

        #out shape (bs, out_dim, seq_len, 1)

        def forward(self, X):
            residual = X
            if self.reshape_residual:
                residual = self.convres(residual)
            X=self.leftpad(x)
            Xa = self.conv3a(self.conv2a(self.conv1a(X)))
            Xb = self.conv3b(self.conv2b(self.conv1b(X)))
            Xb = torch.sigmoid(Xb)
            X = torch.mul(Xa,Xb)
            return X+residual
