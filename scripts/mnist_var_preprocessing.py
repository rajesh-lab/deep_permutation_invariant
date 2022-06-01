import os
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py


class MNISTSetDataset(Dataset):
    def __init__(self, train=True, N=50000, seed=0, min_set_size=6, max_set_size=10, experiment="mnist-var", normalize=False,**kwargs):
        np.random.seed(seed)
        dirpath = 'scripts/mnist_png/'
        if not os.path.exists(dirpath):
            raise NotImplemented("Need to download MNIST")  # https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz

        files_dir = os.path.join(dirpath, "training")
        files_per_label = {i: sorted(glob.glob(os.path.join(files_dir, f'{i}/*.png'))) for i in range(10)}
        if train:
            self.img_filenames = [f for i in range(10) for f in files_per_label[i][:-1000]]
        else:
            self.img_filenames = [f for i in range(10) for f in files_per_label[i][-1000:]]
        self.N = N
        self.min_set_size = min_set_size
        self.max_set_size = max_set_size
        self.experiment = experiment
        self.normalize = normalize

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        num_images = np.random.choice(range(self.min_set_size, self.max_set_size + 1))
        img_filenames = np.random.choice(self.img_filenames, num_images)
        imgs = []
        labels = []
        for filename in img_filenames:
            labels.append(int(filename.rsplit('/', 1)[0].rsplit('/', 1)[1]))
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype('int32')
            img = np.expand_dims(img, 0)
            if self.normalize: 
                img = 2.*(img - np.min(img))/np.ptp(img)-1
            assert img.shape == (1, 28, 28), f"img shape: {img.shape}"
            imgs.append(img)
        
        imgs = np.array(imgs)
        if len(imgs) < self.max_set_size:
            imgs = np.concatenate([imgs, np.zeros((self.max_set_size-num_images, 1, 28, 28))], axis=0)
        assert imgs.shape == (self.max_set_size, 1, 28, 28)
        if self.experiment == 'mnist-var':
            output = np.square(np.std(labels))
        elif self.experiment == 'mnist-mean':
            output = np.mean(labels)
            assert output < 9, "mean is not in range"
        elif self.experiment == "mnist-median":
            output = np.median(labels)
            assert output <= 9, f"median is not in range: {output}"
        elif self.experiment == "mnist-count":
            output = len(np.unique(labels))
            assert output <= 10, f"count not in range: {output}"
        return imgs.astype('int32'), np.array([output]), np.array([num_images]).astype('int32')


def create_dataset(dataset):
    dataloader = DataLoader(dataset,
                                batch_size=64,
                                shuffle=False,
                                num_workers=8)
    images = []
    outs = []
    lengths = []
    for img, label, length in dataloader:
        images.append(img.numpy())
        outs.append(label.numpy())
        lengths.append(length.numpy())
    images = np.concatenate(images, axis=0)
    outs = np.concatenate(outs, axis=0)
    lengths = np.concatenate(lengths, axis=0)
    return images, outs, lengths


if __name__ == "__main__":
    for sample_size in [10]:
        train_dataset = MNISTSetDataset(train=True, min_set_size=sample_size, max_set_size=sample_size)
        images, outs, lengths = create_dataset(train_dataset)
        hf = h5py.File(f'datasets/data/MNIST_var_{sample_size}.h5', 'w')
        hf.create_dataset('train_data', data=images)
        hf.create_dataset('train_labels', data=outs)
        hf.create_dataset('train_lengths', data=lengths)

        test_dataset = MNISTSetDataset(train=False, N=1000, min_set_size=sample_size, max_set_size=sample_size)
        images, outs, lengths = create_dataset(test_dataset)
        hf.create_dataset('test_data', data=images)
        hf.create_dataset('test_labels', data=outs)
        hf.create_dataset('test_lengths', data=lengths)
        hf.close()