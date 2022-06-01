import numpy as np
from torch.utils.data import Dataset

class NormalVarDataset(Dataset):
    def __init__(self, N=1000, n_samples=500, n_dim=2, random_state=0):
        self.N = N
        self.n_samples = n_samples
        self.n_dim = n_dim
        
        self.Xs = []
        self.ys = []
        self.rand= np.random.RandomState(random_state)
        for _ in range(N):
            means = self.rand.uniform(-10, 10, size=self.n_dim)
            variances = self.rand.uniform(0, 10, size=self.n_dim)
            cov = np.diag(variances)
            X = self.rand.multivariate_normal(
                    means, cov, size=self.n_samples,
                    check_valid='warn', tol=1e-8)
            self.Xs.append(X)
            
            stds = np.std(X, axis=0)
            self.ys.append(np.square(stds.ravel()))
        self.sample_size = n_samples
        self.Xs = np.array(self.Xs)
        self.ys = np.array(self.ys)

    def __getitem__(self, index):
        ixs = np.arange(self.Xs[index].shape[0])
        ixs = self.rand.shuffle(ixs)
        return self.Xs[index][ixs, :][0], self.ys[index], np.array([self.sample_size])
    
    def __len__(self):
        return self.N


class WeightedAvgDataset(Dataset):
    def __init__(self, N=1000, n_samples=500, 
                n_dim=2, data_type='normal', 
                random_state=0):
        self.N = N
        self.n_samples = n_samples
        self.n_dim = n_dim
        
        self.Xs = []
        self.ys = []
        self.rand= np.random.RandomState(random_state)
        for _ in range(N):
            if data_type == 'normal':
                means = self.rand.uniform(-10, 10, size=self.n_dim)
                cov = np.eye(self.n_dim)
                X = self.rand.multivariate_normal(means, cov, size=self.n_samples, check_valid='warn', tol=1e-8)
            elif data_type == 'categorical':
                proportions = self.rand.dirichlet(np.ones(self.n_dim))
                idx = self.rand.choice(range(self.n_dim), size=self.n_samples, p=proportions)
                X = np.zeros((idx.size, self.n_dim))
                X[np.arange(idx.size),idx] = 1.
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            self.Xs.append(X)
            averages = np.mean(X, axis=0)
            if data_type == "normal":
                scalar = .001
            else:
                scalar = 1
            weights = np.arange(self.n_dim) * scalar
            self.ys.append(np.array([np.sum(averages * weights)]))
       
        self.sample_size = n_samples
        self.Xs = np.array(self.Xs)
        self.ys = np.array(self.ys)

    def __getitem__(self, index):
        ixs = np.arange(self.Xs[index].shape[0])
        ixs = self.rand.shuffle(ixs)
        return self.Xs[index][ixs, :][0], self.ys[index], np.array([self.sample_size])
    
    def __len__(self):
        return self.N


class NormalVarOODDataset(Dataset):
    def __init__(self, N=1000, n_samples=500, n_dim=2, random_state=0, test=False, ood=True):
        self.N = N
        self.n_samples = n_samples
        self.n_dim = n_dim
        
        self.Xs = []
        self.ys = []
        self.rand= np.random.RandomState(random_state)
        for _ in range(N):
            if test and ood:
                means = self.rand.uniform(-30, 30, size=self.n_dim)
            else:
                means = self.rand.uniform(-10, 10, size=self.n_dim)
            variances = self.rand.uniform(0, 10, size=self.n_dim)
            cov = np.diag(variances)
            X = self.rand.multivariate_normal(
                    means, cov, size=self.n_samples,
                    check_valid='warn', tol=1e-8)
            self.Xs.append(X)
            
            stds = np.mean(np.std(X, axis=0))
            self.ys.append(np.square(stds.ravel()))
        self.sample_size = n_samples
        self.Xs = np.array(self.Xs)
        self.ys = np.array(self.ys)

    def __getitem__(self, index):
        ixs = np.arange(self.Xs[index].shape[0])
        ixs = self.rand.shuffle(ixs)
        return self.Xs[index][ixs, :][0], self.ys[index], np.array([self.sample_size])
    
    def __len__(self):
        return self.N


