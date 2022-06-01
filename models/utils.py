import torch 
import torch.nn as nn

def mask_matrix(matrix, lengths):
    """
    lengths is [batch, 1]

    """
    if lengths is None:
        return matrix
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert lengths.shape == (matrix.shape[0], 1), f"{lengths.shape} vs. {(matrix.shape[0], 1)}"
    batch, n_samples, n_feats = matrix.shape
    # [batch, n_samples]
    length_mask = torch.arange(n_samples).expand(batch, n_samples).to(device) < lengths
    return matrix * length_mask.unsqueeze(-1)


def reshape_x_and_lengths(x, lengths, device):
    x = x.to(device)
    if lengths is None:
        # print('creating lengths')
        lengths = x.shape[1] * torch.ones((x.shape[0], 1)).to(device)
    else:
        lengths = lengths.reshape(x.shape[0], 1)
    assert lengths.shape == (x.shape[0],1), f"lengths should be shaped [batch, n_dist]: {lengths.shape} vs. {(x.shape[0],1)}"
    return x, lengths


def aggregation(x, lengths, input_shape, device, type='mean', categorical=False):
    """
    x: [batch, sample_size, hidden_units]
    (due to the concatenation of the individual encoder outputs)
    lengths: [batch, 1]

    """
    if categorical:
        batch, n_samples = input_shape
    else:
        batch, n_samples, _ = input_shape
    x = x.reshape(batch, n_samples, -1)
    # [batch, n_samples]    
    length_mask = torch.arange(n_samples).expand(lengths.shape[0], 1, n_samples).to(device) < lengths.unsqueeze(-1)
    length_mask = length_mask.squeeze()
    if type == 'sum':
        out = (x * length_mask.unsqueeze(-1)).sum(dim=-2)
    elif type == 'mean':
        # numerator is [batch, n_dists, hidden_units]
        # denominator is [batch, n_dists, 1]
        out = (x * length_mask.unsqueeze(-1)).sum(dim=-2) / length_mask.sum(dim=-1).unsqueeze(-1)
    elif type == 'max':
        length_mask = (1-length_mask.type(torch.FloatTensor).to(device))#*
        length_mask[length_mask!=0] = -float("Inf")
        out = (x+length_mask.unsqueeze(-1)).max(dim=-2)[0]
    else:
        raise ValueError(f"Unsupported type aggregation: {type}")

    out = out.reshape(batch, -1)
    assert len(out.shape) == 2

    if torch.all(torch.eq(lengths, n_samples)):
        if type == 'mean':
            assert torch.allclose(out, x.mean(dim=1).reshape(batch, -1), rtol=1e-05, atol=1e-05), f"aggregation is off: {out} vs. {x.mean(dim=2).reshape(batch, -1)}"
        elif type == 'sum':
            assert torch.allclose(out, x.sum(dim=1).reshape(batch, -1), rtol=1e-05, atol=1e-05), f"aggregation is off: {out} vs. {x.sum(dim=2).reshape(batch, -1)}"
        elif type == 'max':
            assert torch.allclose(out, x.max(dim=1)[0].reshape(batch, -1), rtol=1e-05, atol=1e-05), f"aggregation is off: {out} vs. {x.max(dim=1).reshape(batch, -1)}"
    return out


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                print('len(inputs)', len(inputs))
                inputs = module(inputs)
        return inputs
    
class MyLinear(nn.Linear):
    def forward(self, x, lengths):
        return super().forward(x), lengths