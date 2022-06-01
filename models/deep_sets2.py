import torch
import torch.nn as nn

from models.norms import get_norm
from models.utils import reshape_x_and_lengths, aggregation, MyLinear, MySequential

class MLPHeBase(nn.Module):
    def __init__(self, in_features, out_features, norm="none", sample_size=None):
        super(MLPHeBase, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, 
                            bias=(norm=="none"))
        self.activation = nn.ReLU()
        self.norm = get_norm(norm, sample_size=sample_size, dim_V=in_features)
        self.fc2 = nn.Linear(in_features=out_features,
                            out_features=out_features,
                            bias=(norm=="none"))
        self.norm2 = get_norm(norm, sample_size=sample_size, dim_V=out_features)
        
    def forward(self, X, lengths):
        O = X if getattr(self, 'norm', None) is None else self.norm(X, lengths)[0]
        O = self.activation(O)
        O = self.fc(O)
        O = O if getattr(self, 'norm2', None) is None else self.norm2(O, lengths)[0]
        O = self.activation(O)
        O = self.fc2(O)
        return O

class MLPHe(MLPHeBase):
    def forward(self, X, lengths):
        O = super().forward(X, lengths)
        O = X + O
        return O, lengths


class MLPAgg(MLPHeBase):
    def forward(self, X, lengths):
        O = super().forward(X, lengths)
        # [0] is for max since max outputs tuple
        # O = torch.max(X, -2, keepdim=True)[0] + O
        O = torch.mean(X, -2, keepdim=True) + O
        return O, lengths


class MLPResNet(nn.Module):
    def __init__(self, in_features, out_features, norm="none", sample_size=None):
        super(MLPResNet, self).__init__()
        self.fc = nn.Linear(in_features=in_features,
                            out_features=out_features, bias=(
                            norm=="none"))
        self.activation = nn.ReLU()
        self.norm = get_norm(norm, sample_size=sample_size, dim_V=out_features)
        self.fc2 = nn.Linear(in_features=out_features,
                             out_features=out_features,
                             bias=(norm=="none"))
        self.norm2 = get_norm(norm, sample_size=sample_size, dim_V=out_features)
        
    
    def forward(self, X, lengths):
        O = self.fc(X)
        O = O if getattr(self, 'norm', None) is None else self.norm(O, lengths)[0]
        O = self.activation(O)
        O = self.fc2(O)
        O = O if getattr(self, 'norm2', None) is None else self.norm2(O, lengths)[0] 
        O = X + O
        O = self.activation(O)
        return O, lengths
            

class DeepSets2Base(nn.Module):
    """
    Basic class, should not be used by itself
    """
    
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000, 
                 res_pipeline='he',aggregation_type='sum'):
        super(DeepSets2Base, self).__init__()
        self.aggregation_type = aggregation_type
        layers = [MyLinear(n_inputs, dim_hidden, bias=(norm=="none"))]
        if res_pipeline == 'resnet':
            MLP = MLPResNet
        elif res_pipeline == 'he':
            MLP = MLPHe
        elif res_pipeline == "agg":
            MLP = MLPAgg

        for j in range(n_enc_layers):
            layers.append(MLP(in_features=dim_hidden, out_features=dim_hidden,
                            norm=norm,sample_size=sample_size))
        self.enc = MySequential(*layers)
        if res_pipeline == 'he':
            self.norm = get_norm(norm, sample_size=sample_size, dim_V=dim_hidden)
            self.fc = nn.Sequential(nn.ReLU(), 
                                    nn.Linear(dim_hidden, dim_hidden))
        else:
            self.fc = nn.Linear(dim_hidden, dim_hidden)
        self.dec = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(),
                                 nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(), 
                                 nn.Linear(dim_hidden, dim_hidden),
                                 nn.ReLU(),
                                 nn.Linear(dim_hidden, n_outputs))

    def forward(self, x, lengths): 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, lengths = reshape_x_and_lengths(x, lengths, device)
        input_shape = x.shape
        out,_ = self.enc.forward(x, lengths)
        out = out if getattr(self, 'norm', None) is None else self.norm(out, lengths)[0]
        out = self.fc(out)
        out = aggregation(out, lengths, input_shape, device,
                         type=self.aggregation_type)
        return self.dec(out)


class DeepSets2Sum(DeepSets2Base):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000,
                 res_pipeline='he'):
        super(DeepSets2Sum, self).__init__(n_inputs, n_outputs, 
                            n_enc_layers, dim_hidden, norm,
                            sample_size, res_pipeline, 'sum')
    
    def forward(self, x, lengths=None): 
        return super().forward(x, lengths)
        


class DeepSets2Max(DeepSets2Base):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000,
                 res_pipeline='he'):
        super(DeepSets2Max, self).__init__(n_inputs, n_outputs, 
                            n_enc_layers, dim_hidden, norm,
                            sample_size, res_pipeline, 'max')
    
    def forward(self, x, lengths=None): 
        return super().forward(x, lengths)