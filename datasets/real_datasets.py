import numpy as np
from torch.utils.data import Dataset
import h5py
import sys

def standardize(x):
    clipper = np.mean(np.abs(x), (0, 1), keepdims=True)
    z = np.clip(x, -100 * clipper, 100 * clipper)
    mean = np.mean(z, (0, 1), keepdims=True)
    std = np.std(z, (0, 1), keepdims=True)
    return (z - mean) / std


class PointCloudDataset(Dataset):
    def __init__(self, test=False, do_standardize=True, sample_size=1000, random_state=0, categorical=False):

        if categorical:
            with h5py.File("datasets/data/ModelNet40_cloud_categorical.h5", 'r') as f:
                self._train_data = np.array(f['train_data']).astype(np.int32)
                self._train_label = np.array(f['train_labels'])
                self._test_data = np.array(f['test_data']).astype(np.int32)
                self._test_label = np.array(f['test_labels'])
        else:
            with h5py.File("datasets/data/ModelNet40_cloud.h5", 'r') as f:
                self._train_data = np.array(f['tr_cloud'])
                self._train_label = np.array(f['tr_labels'])
                self._test_data = np.array(f['test_cloud'])
                self._test_label = np.array(f['test_labels'])
        
        self.rand= np.random.RandomState(random_state)
        self.test = test
        self.sample_size = sample_size
        self.num_classes = np.max(self._train_label) + 1
        self.prep = standardize if do_standardize else lambda x: x
        self.perm = self.rand.permutation(self._train_data.shape[1])[:sample_size]

    def __getitem__(self, index):
        perm = self.perm
        if self.test:
            return self.prep(self._test_data[index, perm]), self._test_label[index], np.array([self.sample_size])
        else:
            return self.prep(self._train_data[index, perm]), self._train_label[index], np.array([self.sample_size])
        
    def __len__(self):
        return len(self._test_data) if self.test else len(self._train_data)


class HematocritDataset(Dataset):
    def __init__(self, test=False, sample_size=1000, random_state=0):

        with h5py.File("datasets/data/hematocrit.h5", 'r') as f:
            self._train_data = np.array(f['train_data'])
            self._train_label = np.array(f['train_labels'])
            self._test_data = np.array(f['test_data'])
            self._test_label = np.array(f['test_labels'])
        
        self.sample_size = sample_size
        self.test = test
        self.rand= np.random.RandomState(random_state)
        self.perm = self.rand.permutation(self._train_data.shape[1])[:sample_size]
    
    def __getitem__(self, index):
        perm = self.perm

        if self.test:
            return self._test_data[index, perm], self._test_label[index], np.array([self.sample_size])
        else:
            return self._train_data[index, perm], self._train_label[index], np.array([self.sample_size])
        
    def __len__(self):
        return len(self._test_data) if self.test else len(self._train_data)
 


class HematocritCategoricalDataset(Dataset):
    def __init__(self, test=False, sample_size=1000, categorical=True, random_state=0):
        
        if categorical:
            with h5py.File("datasets/data/hematocrit_categorical.h5", 'r') as f:
                self._train_data = np.array(f['train_data'])
                self._train_label = np.array(f['train_labels'])
                self._test_data = np.array(f['test_data'])
                self._test_label = np.array(f['test_labels'])
        else:
            with h5py.File("datasets/data/hematocrit_small.h5", 'r') as f:
                self._train_data = np.array(f['train_data'])
                self._train_label = np.array(f['train_labels'])
                self._test_data = np.array(f['test_data'])
                self._test_label = np.array(f['test_labels'])
        self.sample_size = sample_size
        self.test = test
        self.rand= np.random.RandomState(random_state)
        self.perm = self.rand.permutation(self._train_data.shape[1])[:sample_size]
    
    def __getitem__(self, index):
        perm = self.perm

        if self.test:
            return self._test_data[index, perm], self._test_label[index], np.array([self.sample_size])
        else:
            return self._train_data[index, perm], self._train_label[index], np.array([self.sample_size])
        
    def __len__(self):
        return len(self._test_data) if self.test else len(self._train_data)
    

class MNISTVarDataset(Dataset):
    def __init__(self, test=False,sample_size=10, random_state=0):
        filenames = {10:'datasets/data/MNIST_var_10.h5',
                     100: 'datasets/data/MNIST_var_100.h5',
                     1000: 'datasets/data/MNIST_var_1000.h5'}

        with h5py.File(filenames[sample_size], 'r') as f:
            self._train_data = np.array(f['train_data'])
            self._train_label = np.array(f['train_labels'])
            self._test_data = np.array(f['test_data'])
            self._test_label = np.array(f['test_labels'])
        
        self.sample_size = sample_size
        self.test = test
        self.rand= np.random.RandomState(random_state)
        self.perm = self.rand.permutation(self._train_data.shape[1])
    
    def __getitem__(self, index):
        perm = self.perm
        if self.test:
            return self._test_data[index, perm], self._test_label[index], np.array([self.sample_size])
        else:
            return self._train_data[index, perm], self._train_label[index], np.array([self.sample_size])
        
    def __len__(self):
        return len(self._test_data) if self.test else len(self._train_data)


class MNISTVarDatasetDiffLengths(Dataset):
    def __init__(self, test=False, random_state=0):
        with h5py.File('datasets/data/MNIST_var_6_10.h5', 'r') as f:
            self._train_data = np.array(f['train_data'])
            self._train_label = np.array(f['train_labels'])
            self._test_data = np.array(f['test_data'])
            self._test_label = np.array(f['test_labels'])
            self._train_length = np.array(f['train_lengths'])
            self._test_length = np.array(f['test_lengths'])
        
        self.test = test
        self.rand= np.random.RandomState(random_state)
        self.perm = self.rand.permutation(self._train_data.shape[1])
    
    def __getitem__(self, index):
        perm = self.perm
        if self.test:
            return self._test_data[index, perm], self._test_label[index], self._test_length[index]
        else:
            return self._train_data[index, perm], self._train_label[index], self._train_length[index]
        
    def __len__(self):
        return len(self._test_data) if self.test else len(self._train_data)
    
    
class CelebAAnomalyDetection(Dataset):
    def __init__(self, test=False,sample_size=10, random_state=0):
        if test:
            with h5py.File("datasets/data/CelebA_test_10.h5", 'r') as f:
                self.data = np.array(f['test_data'])
                self.label = np.array(f['test_labels'])
        else:
            with h5py.File("datasets/data/CelebA_train_10.h5", 'r') as f:
                self.data = np.array(f['train_data'])
                self.label = np.array(f['train_labels'])
        
       
    def __getitem__(self, index):
        return self.data[index], self.label[index], np.array([10])
        
    def __len__(self):
        return len(self.data)


class AnemiaDataset(Dataset):
    def __init__(self, test=False,sample_size=100, random_state=0):
        if test:
            with h5py.File("datasets/data/Anemia.h5", 'r') as f:
                self.data = np.array(f['test_data'])
                self.label = np.array(f['test_labels'])
        else:
            with h5py.File("datasets/data/Anemia.h5", 'r') as f:
                self.data = np.array(f['train_data'])
                self.label = np.array(f['train_labels'])
        print(np.unique(self.label))
        self.sample_size = sample_size
        self.rand= np.random.RandomState(random_state)
        self.perm = self.rand.permutation(self.data.shape[1])[:sample_size]
       
    def __getitem__(self, index):
        return self.data[index][0], self.label[index], np.array([self.sample_size])
        
    def __len__(self):
        return len(self.data)

