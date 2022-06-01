import torch
import torch.nn as nn
import math 
import numpy as np 

import torch.nn.functional as F

from models.norms import get_norm
from models.utils import mask_matrix, reshape_x_and_lengths, MySequential, MyLinear


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, 
                 norm='ln1_fp', v_norm_samples=32,
                 sample_size=1000, normalizeQ=True):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.normalizeQ = normalizeQ
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.normQ = get_norm(norm, sample_size=sample_size, dim_V=dim_Q)
        self.normK = get_norm(norm, sample_size=v_norm_samples, dim_V=dim_K)
        self.norm0 = get_norm(norm, sample_size=sample_size, dim_V=dim_V)
        
    def forward(self, Q, K, lengths=None, mask=[]):
        _input = Q
        if self.normalizeQ:
            Q = Q if getattr(self, 'normQ', None) is None else self.normQ(Q, lengths)[0]
        K = K if getattr(self, 'normK', None) is None else self.normK(K, lengths)[0]
        
        Q = self.fc_q(Q)
        if "Q" in mask:
            Q = mask_matrix(Q, lengths)
        
        K, V = self.fc_k(K), self.fc_v(K)
        if "K" in mask:
            K = mask_matrix(K, lengths)
            V = mask_matrix(V, lengths)

        dim_split = self.dim_V // self.num_heads
        dim_split = 2**int(round(np.log2(dim_split),0))
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        _input, _ = torch.max(_input, -2, keepdim=True)  # mean over samples
        input_multihead = torch.cat(_input.split(dim_split, 2), 0)
        O = torch.cat((input_multihead + A.bmm(V_)).split(Q.size(0), 0), 2)
        normed_O = O if getattr(self, 'norm0', None) is None else self.norm0(O, lengths)[0]
        O, _ = torch.max(O, -2, keepdim=True)  # mean over samples
        O = O + F.relu(self.fc_o(normed_O))
        return O

    
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, sample_size=100, norm="none"):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, norm=norm,
                        v_norm_samples=sample_size, sample_size=num_inds,
                        normalizeQ=False)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, norm=norm,
                        v_norm_samples=num_inds, sample_size=sample_size,
                        normalizeQ=True)

    def forward(self, X, lengths=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, lengths=lengths, mask=["K"])
        return self.mab1(X, H, lengths=lengths, mask=["Q"]), lengths


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, sample_size=1, norm='none'):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads,
                        sample_size=sample_size, norm=norm)

    def forward(self, X, lengths=None):
        return self.mab(X, X, lengths=lengths, mask=['Q', 'K']), lengths


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, norm='none', sample_size=1)

    def forward(self, X, lengths=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, lengths=lengths, mask=["K"]), lengths


class SetTransformer2Agg(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000):
        super(SetTransformer2Agg, self).__init__()

        num_heads = 4 
        num_inds = 32 

        layers = [MyLinear(n_inputs, dim_hidden)]
        for i in range(n_enc_layers):  
            layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, norm=norm, sample_size=sample_size))
        if norm != "none":
            layers.append(get_norm(norm, sample_size=sample_size, dim_V=dim_hidden))
        self.enc = MySequential(*layers)
        self.dec = MySequential(
            PMA(dim_hidden, num_heads, 1),
            SAB(dim_hidden, dim_hidden, num_heads, norm="none", sample_size=1),
            SAB(dim_hidden, dim_hidden, num_heads, norm="none", sample_size=1),
            SAB(dim_hidden, dim_hidden, num_heads, norm="none", sample_size=1),
            MyLinear(dim_hidden, n_outputs)
        )

    def forward(self, x, lengths):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, lengths = reshape_x_and_lengths(x, lengths, device)
        out, _= self.enc(x, lengths)
        out = self.dec(out, lengths)[0].squeeze()
        return out