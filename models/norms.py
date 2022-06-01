import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNorm(nn.BatchNorm1d):
    def forward(self, x, lengths=None):
        # [batch, features] after aggregation, [batch, samples, features] otherwise
        orig_dim = x.dim()
        orig_shape = x.shape
        assert orig_dim == 2 or orig_dim == 3, f"Initial input should be 2D or 3D, got {orig_dim}D"
        if orig_dim == 2:
            x = x.unsqueeze(-1)
        x = torch.transpose(x, 1, 2)  # [batch, features, samples]
        out = super().forward(x)
        if orig_dim == 2:
            out = out.squeeze(1)
        else:
            out = torch.transpose(out, 1, 2)  # [batch, samples, features]
        assert out.shape == orig_shape, f"Input should be {orig_shape}, got {out.shape}"
        return out, lengths


class FeatureNormL(nn.BatchNorm1d):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)
        # we don't implement running means and variances for now
        # self.running_means = None
        # self.running_vars = None

    def forward(self, x, lengths=None):
        # standardization
        batch, n_samples, n_features = x.shape
        x = x.reshape(batch, n_samples, -1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        length_mask = torch.arange(n_samples).expand(lengths.shape[0], n_samples).to(device) < lengths
        means_for_each_set = (x * length_mask.unsqueeze(-1)).sum(dim=1)/length_mask.sum(dim=1)[:, None]
        std_for_each_set = ((x-means_for_each_set[:, None, :]).square() * length_mask.unsqueeze(-1)).sum(dim=1) /length_mask.sum(dim=1)[:, None]
        means = means_for_each_set.mean(axis=0)
        std = std_for_each_set.mean(axis=0)
        std = torch.sqrt(std + self.eps)
        
        out = (x-means[None, None,:])/std[None, None,:]
        
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out, lengths


class FeatureNormL1(nn.BatchNorm1d):
    """ Averages on sample level, so different sets have different contributions """
    def __init__(self, feature_dim, **kwargs):
        super().__init__(feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)
        # we don't implement running means and variances for now
        # self.running_means = None
        # self.running_vars = None

    def forward(self, x, lengths=None):
        # standardization
        batch, n_samples, n_features = x.shape
        x = x.reshape(batch, n_samples, -1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        length_mask = torch.arange(n_samples).expand(lengths.shape[0], n_samples).to(device) < lengths
        means = (x * length_mask.unsqueeze(-1)).sum(dim=[0,1])/length_mask.sum(dim=[0,1])
        stds = ((x-means[None, None, :]).square() * length_mask.unsqueeze(-1)).sum(dim=[0,1])/length_mask.sum(dim=[0,1])
        std = torch.sqrt(stds + self.eps)
        
        out = (x-means[None, None,:])/std[None, None,:]
        
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out, lengths


class SetNorm(nn.LayerNorm):
    def __init__(self, *args, feature_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x, lengths=None):
        # standardization
        out = super().forward(x)
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out, lengths


class SetNormL(nn.LayerNorm):     
    def __init__(self, *args, feature_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x, lengths=None):
        # standardization
        batch, n_samples, n_features = x.shape
        x = x.reshape(batch, n_samples, -1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        length_mask = torch.arange(n_samples).expand(lengths.shape[0], n_samples).to(device) < lengths
        means = (x * length_mask.unsqueeze(-1)).sum(dim=[1, 2]) / (length_mask.sum(dim=1)*n_features)
        means = means.reshape(batch, 1)
        std = torch.sqrt(((x-means[:, None]).square() * length_mask.unsqueeze(-1)).sum(dim=[1, 2]) /(length_mask.sum(dim=1)*n_features) + self.eps)
        std = std.reshape(batch, 1)
        out = (x-means[:, None])/std[:, None]
        
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out, lengths


class LayerNormL(nn.LayerNorm):
    def __init__(self, feature_dim, *args, **kwargs):
        super().__init__(feature_dim, *args, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x, lengths=None):
        # standardization
        batch, n_samples, n_features = x.shape
        x = x.reshape(batch, n_samples, -1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        length_mask = torch.arange(n_samples).expand(lengths.shape[0], n_samples).to(device) < lengths
        means = (x * length_mask.unsqueeze(-1)).sum(dim=2) /  n_features
        std = torch.sqrt(((x-means.unsqueeze(-1)).square() * length_mask.unsqueeze(-1)).sum(dim=2) / n_features + self.eps)
        out = (x-means.unsqueeze(-1))/std.unsqueeze(-1)
        
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out, lengths


class InstanceNorm(nn.LayerNorm):
    def forward(self, x, lengths=None):
        return super().forward(x), lengths


class LayerNorm(nn.LayerNorm):
    def forward(self, x, lengths=None):
        return super().forward(x), lengths

def get_norm(block_norm, sample_size, dim_V):
    if block_norm == "layer_norm":
        return LayerNorm(dim_V)
    elif block_norm == "feature_norm":
        return FeatureNorm(dim_V)
    elif block_norm == "set_norm":
        return SetNorm([sample_size, dim_V], elementwise_affine=False, feature_dim=dim_V)
    elif block_norm == "layer_norml":
        return LayerNormL(dim_V, elementwise_affine=False)  # we define the transformation params
    elif block_norm == "feature_norml":
        return FeatureNormL(feature_dim=dim_V, affine=False)
    elif block_norm == "set_norml":
        return SetNormL([sample_size, dim_V], elementwise_affine=False, feature_dim=dim_V)
    elif block_norm == "feature_norml1":
        return FeatureNormL1(feature_dim=dim_V, affine=False)
    return None
