import torch
import torch.nn as nn
import math 
import numpy as np 

from models.deep_sets import DeepSetsBase
from models.deep_sets2 import DeepSets2Base
from models.set_transformer import SetTransformer
from models.set_transformer2 import SetTransformer2
from models.utils import reshape_x_and_lengths

class SetTransformer2Conv(SetTransformer2):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000):
        
        super(SetTransformer2Conv, self).__init__(n_inputs=256, n_outputs=n_outputs,
                                             n_enc_layers=n_enc_layers, dim_hidden=dim_hidden, 
                                             norm = norm, sample_size = sample_size)
        self.featurizer = nn.Sequential(nn.Conv2d(3,32, kernel_size=3, bias=True),
                                        nn.Conv2d(32,32, kernel_size=3, bias=True),
                                        nn.Conv2d(32,64, kernel_size=3,  bias=True),
                                         nn.MaxPool2d(kernel_size=2),
                                         nn.Conv2d(64,64, kernel_size=3, bias=True),
                                        nn.Conv2d(64,64, kernel_size=3, bias=True),
                                        nn.Conv2d(64,128, kernel_size=3,  bias=True),
                             nn.MaxPool2d(kernel_size=2),
                            nn.Conv2d(128,128, kernel_size=3, bias=True),
                                        nn.Conv2d(128,128, kernel_size=3, bias=True),
                                        nn.Conv2d(128,256, kernel_size=3,  bias=True),
                           nn.MaxPool2d(kernel_size=5),
                            )
                                        
       
        self.final_layer = nn.Linear(dim_hidden, 1)
     
    def forward(self, x, lengths=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, lengths = reshape_x_and_lengths(x, lengths, device)
        out = x.transpose(4, 2)
        B, S, C, H, W = out.shape
        out = out.view(-1, C, H, W)
        out = self.featurizer(out)
        _,  C1, H1, W1 = out.shape
        out = out.view(B, S, C1, H1, W1)
        out = out.squeeze()
        out, _= self.enc(out, lengths)
        out = self.final_layer(out).squeeze()
        return nn.Softmax()(out)
        

class SetTransformerConv(SetTransformer):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000):
        
        super(SetTransformerConv, self).__init__(n_inputs=256, n_outputs=n_outputs,
                                             n_enc_layers=n_enc_layers, dim_hidden=dim_hidden, 
                                             norm = norm, sample_size = sample_size)
        self.featurizer = nn.Sequential(nn.Conv2d(3,32, kernel_size=3, bias=True),
                                        nn.Conv2d(32,32, kernel_size=3, bias=True),
                                        nn.Conv2d(32,64, kernel_size=3,  bias=True),
                                         nn.MaxPool2d(kernel_size=2),
                                         nn.Conv2d(64,64, kernel_size=3, bias=True),
                                        nn.Conv2d(64,64, kernel_size=3, bias=True),
                                        nn.Conv2d(64,128, kernel_size=3,  bias=True),
                             nn.MaxPool2d(kernel_size=2),
                            nn.Conv2d(128,128, kernel_size=3, bias=True),
                                        nn.Conv2d(128,128, kernel_size=3, bias=True),
                                        nn.Conv2d(128,256, kernel_size=3,  bias=True),
                           nn.MaxPool2d(kernel_size=5),
                            )
                                        
      
        self.final_layer = nn.Linear(dim_hidden, 1)
     
    def forward(self, x, lengths=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, lengths = reshape_x_and_lengths(x, lengths, device)
        out = x.transpose(4, 2)
        B, S, C, H, W = out.shape
        out = out.view(-1, C, H, W)
        out = self.featurizer(out)
        _,  C1, H1, W1 = out.shape
        out = out.view(B, S, C1, H1, W1)
        out = out.squeeze()
        out, _= self.enc(out, lengths)
        out = self.final_layer(out).squeeze()
        return nn.Softmax()(out)
    
    
class DeepSets2Conv(DeepSets2Base):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000,
                  res_pipeline='he',aggregation_type='sum'):
        super(DeepSets2Conv, self).__init__(n_inputs=256, n_outputs=n_outputs,
                                             n_enc_layers=n_enc_layers, dim_hidden=dim_hidden, 
                                             norm = norm, sample_size = sample_size,
                                           res_pipeline=res_pipeline,
                                           aggregation_type=aggregation_type)
        self.featurizer = nn.Sequential(nn.Conv2d(3,32, kernel_size=3, bias=True),
                                        nn.Conv2d(32,32, kernel_size=3, bias=True),
                                        nn.Conv2d(32,64, kernel_size=3,  bias=True),
                                         nn.MaxPool2d(kernel_size=2),
                                         nn.Conv2d(64,64, kernel_size=3, bias=True),
                                        nn.Conv2d(64,64, kernel_size=3, bias=True),
                                        nn.Conv2d(64,128, kernel_size=3,  bias=True),
                             nn.MaxPool2d(kernel_size=2),
                            nn.Conv2d(128,128, kernel_size=3, bias=True),
                                        nn.Conv2d(128,128, kernel_size=3, bias=True),
                                        nn.Conv2d(128,256, kernel_size=3,  bias=True),
                           nn.MaxPool2d(kernel_size=5),
                            )
                                        
       
        self.final_layer = nn.Linear(dim_hidden, 1)
     
    def forward(self, x, lengths=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, lengths = reshape_x_and_lengths(x, lengths, device)
        out = x.transpose(4, 2)
        B, S, C, H, W = out.shape
        out = out.view(-1, C, H, W)
        out = self.featurizer(out)
        _, C1, H1, W1 = out.shape
        out = out.view(B, S, C1, H1, W1)
        out = out.squeeze()
        out, _= self.enc(out, lengths)
        out = self.final_layer(out).squeeze()
        return nn.Softmax()(out)
    
    
class DeepSetsConv(DeepSetsBase):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000,aggregation_type='sum'):
        
        super(DeepSetsConv, self).__init__(n_inputs=256, n_outputs=n_outputs,
                                             n_enc_layers=n_enc_layers, dim_hidden=dim_hidden, 
                                             norm = norm, sample_size = sample_size,
                                           aggregation_type=aggregation_type)
        self.featurizer = nn.Sequential(nn.Conv2d(3,32, kernel_size=3, bias=True),
                                        nn.Conv2d(32,32, kernel_size=3, bias=True),
                                        nn.Conv2d(32,64, kernel_size=3,  bias=True),
                                         nn.MaxPool2d(kernel_size=2),
                                         nn.Conv2d(64,64, kernel_size=3, bias=True),
                                        nn.Conv2d(64,64, kernel_size=3, bias=True),
                                        nn.Conv2d(64,128, kernel_size=3,  bias=True),
                             nn.MaxPool2d(kernel_size=2),
                            nn.Conv2d(128,128, kernel_size=3, bias=True),
                                        nn.Conv2d(128,128, kernel_size=3, bias=True),
                                        nn.Conv2d(128,256, kernel_size=3,  bias=True),
                           nn.MaxPool2d(kernel_size=5),
                            )
                                        
        
        self.final_layer = nn.Linear(dim_hidden, 1)
        
    def forward(self, x, lengths=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, lengths = reshape_x_and_lengths(x, lengths, device)
        out = x.transpose(4, 2)
        B, S, C, H, W = out.shape
        out = out.view(-1, C, H, W)
        out = self.featurizer(out)
        
        _,  C1, H1, W1 = out.shape
        out = out.view(B, S, C1, H1, W1)
        out = out.squeeze()
        out, _= self.enc(out, lengths)
        out = self.final_layer(out).squeeze()
      
        return nn.Softmax()(out)
        