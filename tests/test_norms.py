import torch
import unittest
from models.norms import get_norm
from experiments.utils import count_parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestNorms(unittest.TestCase):
    def test_ln(self):
        batch_size = 21
        sample_size = 44
        dim_V = 5
        x = torch.rand((batch_size, sample_size, dim_V))
        norm = get_norm('layer_norm', sample_size, dim_V).to(device)
        self.assertEqual(count_parameters(norm), dim_V * 2)  # correct
        # self.assertEqual(count_parameters(norm), 0)  # incorrect
        lengths = (torch.ones(batch_size) * 44).reshape((batch_size, 1)).to(device)
        y, _ = norm(x.to(device), lengths)
        self.assertTrue(torch.all(torch.isclose(torch.mean(y, axis=[2]).cpu(), torch.zeros((batch_size, sample_size)), atol=1e-5)))     
        self.assertTrue(torch.all(torch.isclose(torch.var(y, axis=[2], unbiased=False).cpu(), torch.ones((batch_size, sample_size)), atol=5e-2)))

        norml = get_norm('layer_norml', sample_size, dim_V).to(device)
        self.assertEqual(count_parameters(norml), dim_V * 2)
        yl, _ = norml(x.to(device), lengths)
        self.assertTrue(torch.all(torch.isclose(y, yl, atol=1e-4)))
    
    def test_fn(self):
        batch_size = 21
        sample_size = 44
        dim_V = 5
        x = torch.rand((batch_size, sample_size, dim_V))
        norm = get_norm('feature_norm', sample_size, dim_V).to(device)
        self.assertEqual(count_parameters(norm), dim_V * 2)
        lengths = (torch.ones(batch_size) * 44).reshape((batch_size, 1)).to(device)
        y, _ = norm(x.to(device), lengths)
        self.assertTrue(torch.all(torch.isclose(torch.mean(y, axis=[0, 1]).cpu(), torch.zeros(dim_V), atol=1e-5)))
        self.assertTrue(torch.all(torch.isclose(torch.var(y, axis=[0, 1], unbiased=False).cpu(), torch.ones(dim_V), atol=1e-3)))

        norml = get_norm('feature_norml', sample_size, dim_V).to(device)
        self.assertEqual(count_parameters(norml), dim_V * 2)
        yl, _ = norml(x.to(device), lengths)
    
        self.assertTrue(torch.all(torch.isclose(y, yl, atol=5e-2)))

    def test_fn1(self):
        batch_size = 21
        sample_size = 44
        dim_V = 5
        x = torch.rand((batch_size, sample_size, dim_V))
        norm = get_norm('feature_norml1', sample_size, dim_V).to(device)
        self.assertEqual(count_parameters(norm), dim_V * 2)
        lengths = (torch.ones(batch_size) * 44).reshape((batch_size, 1)).to(device)
        y, _ = norm(x.to(device), lengths)
        self.assertTrue(torch.all(torch.isclose(torch.mean(y, axis=[0, 1]).cpu(), torch.zeros(dim_V), atol=1e-5)))
        self.assertTrue(torch.all(torch.isclose(torch.var(y, axis=[0, 1], unbiased=False).cpu(), torch.ones(dim_V), atol=1e-3)))
    
    def test_sn(self):
        batch_size = 21
        sample_size = 44
        dim_V = 5
        x = torch.rand((batch_size, sample_size, dim_V))
        norm = get_norm('set_norm', sample_size, dim_V).to(device)
        self.assertEqual(count_parameters(norm), dim_V * 2)
        lengths = (torch.ones(batch_size) * 44).reshape((batch_size, 1)).to(device)
        y, _ = norm(x.to(device), lengths)
        self.assertTrue(torch.all(torch.isclose(torch.mean(y, axis=[1, 2]).cpu(), torch.zeros(batch_size), atol=1e-5)))
        self.assertTrue(torch.all(torch.isclose(torch.var(y, axis=[1, 2], unbiased=False).cpu(), torch.ones(batch_size), atol=1e-3)))

        norml = get_norm('set_norml', sample_size, dim_V).to(device)
        self.assertEqual(count_parameters(norml), dim_V * 2)
        yl, _ = norml(x.to(device), lengths)
    
        self.assertTrue(torch.all(torch.isclose(y, yl, atol=1e-5)))


