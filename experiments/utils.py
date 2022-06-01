import torch
import numpy as np
import random
from datasets import HematocritDataset, PointCloudDataset, \
            MNISTVarDataset, NormalVarDataset, MNISTVarDatasetDiffLengths,AnemiaDataset, HematocritCategoricalDataset


N_INPUTS = {'pointcloud':3,
            'pointcloud_categorical':1000, 
            'hematocrit':2, 
            'normalvar':1, 
            'mnistvar':784,
            'mnistvardiff':784,
            'anemia':2, 
            'hematocrit_categorical':100, 
            'hematocrit_small':2}
N_OUTPUTS = {'pointcloud':40, 
                'pointcloud_categorical':40,
                'hematocrit':1, 
                'normalvar':1, 
                'mnistvar':1,
                'mnistvardiff':1,
            'anemia':2, 
            'hematocrit_categorical':1, 
            'hematocrit_small':1}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataset(task):
    if task == 'hematocrit':
        train = HematocritDataset(test=False, sample_size=1000)
        test = HematocritDataset(test=True, sample_size = 1000)
    elif task == 'anemia':
        train = AnemiaDataset(test=False, sample_size=1000)
        test = AnemiaDataset(test=True, sample_size = 1000)
    elif task == 'pointcloud':
        train = PointCloudDataset(test=False, sample_size=1000, 
                                 do_standardize=True)
        test = PointCloudDataset(test=True, sample_size=1000, 
                                 do_standardize=True)
    elif task == 'pointcloud_categorical':
        train = PointCloudDataset(test=False, sample_size=1000, 
                                 do_standardize=False, categorical=True)
        test = PointCloudDataset(test=True, sample_size=1000, 
                                 do_standardize=False, categorical=True)
    elif task == 'mnistvar':
        train = MNISTVarDataset(test=False, sample_size=10)
        test = MNISTVarDataset(test=True, sample_size=10)
    elif task == 'mnistvardiff':
        train = MNISTVarDatasetDiffLengths(test=False)
        test = MNISTVarDatasetDiffLengths(test=True)
    elif task == 'normalvar':
        train = NormalVarDataset(N=10000, n_samples=1000,
                                 n_dim=1, random_state=0)
        test = NormalVarDataset(N=1000, n_samples=1000,
                                 n_dim=1, random_state=1)
    elif task == 'hematocrit_categorical':
        train = HematocritCategoricalDataset(test=False, sample_size=1000, categorical=True)
        test = HematocritCategoricalDataset(test=True, sample_size = 1000, categorical=True)
    elif task == 'hematocrit_small':
        train = HematocritCategoricalDataset(test=False, sample_size=1000, categorical=False)
        test = HematocritCategoricalDataset(test=True, sample_size = 1000, categorical=False)
    else:
        raise ValueError(f"Unknown task: {task}")
    return train, test

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
