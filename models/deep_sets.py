import torch
import torch.nn as nn

from models.norms import get_norm
from models.utils import reshape_x_and_lengths, aggregation, MyLinear, MySequential

class MLP(nn.Module):
    def __init__(self, in_features, out_features, norm="none", sample_size=None):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=(norm=="none"))
        self.norm = get_norm(norm, sample_size=sample_size, dim_V=out_features)
        self.activation = nn.ReLU()
    
    def forward(self, X, lengths):
        O = self.fc(X)
        O,_ = (O, lengths) if getattr(self, 'norm', None) is None else self.norm(O, lengths)
        O = self.activation(O)
        return O, lengths

class Embedding(nn.Embedding):
    def forward(self, X, lengths):
        O = super().forward(X)
        return O, lengths

class DeepSetsBase(nn.Module):
    """
        Basic class, should not be used by itself
    
    """
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000, 
                 aggregation_type='sum', categorical=False):
        super(DeepSetsBase, self).__init__()
        self.aggregation_type = aggregation_type
        if categorical:
            layers = [Embedding(1000, dim_hidden)]
        else:
            layers = [MLP(in_features=n_inputs, out_features=dim_hidden,
                        norm=norm, sample_size=sample_size)]

        for j in range(n_enc_layers-1):
            layers.append(MLP(in_features=dim_hidden, out_features=dim_hidden,
                             norm=norm,sample_size=sample_size))

        layers.append(MyLinear(in_features=dim_hidden, out_features=dim_hidden))
        self.enc = MySequential(*layers)
        self.dec = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(),
                                 nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(), 
                                 nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(),
                                 nn.Linear(dim_hidden, n_outputs))
        self.categorical = categorical

    def forward(self, x, lengths=None): 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, lengths = reshape_x_and_lengths(x, lengths, device)
        input_shape = x.shape
        out, _ = self.enc.forward(x, lengths)
        out = aggregation(out, lengths, input_shape, device,
                         type=self.aggregation_type, categorical=self.categorical)
        return self.dec(out)


class DeepSetsSum(DeepSetsBase):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000, categorical=False):
        super(DeepSetsSum, self).__init__(n_inputs, n_outputs, 
                            n_enc_layers, dim_hidden, norm,
                            sample_size, 'sum', categorical=categorical)
    
    def forward(self, x, lengths=None): 
        return super().forward(x, lengths)
        


class DeepSetsMax(DeepSetsBase):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000, categorical=False):
        super(DeepSetsMax, self).__init__(n_inputs, n_outputs, 
                            n_enc_layers, dim_hidden, norm,
                            sample_size, 'max', categorical=categorical)
    
    def forward(self, x, lengths=None): 
        return super().forward(x, lengths)


