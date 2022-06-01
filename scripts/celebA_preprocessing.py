import sys
import os
sys.path.insert(0, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib.pyplot import imshow
import h5py

def single_set_creation(data, n_images=10):   
    # select two random attribute for image
    attributes_set = np.random.choice(np.arange(40), 2, replace=False)
    
    #select images with those attributes
    withattribute = attributes_train[(attributes_train.iloc[:,attributes_set[0]]==1)&
                        (attributes_train.iloc[:,attributes_set[1]]==1)]
    
    #select images without those attributes
    withoutattribute = attributes_train[(attributes_train.iloc[:,attributes_set[0]]!=1)&
                            (attributes_train.iloc[:,attributes_set[1]]!=1)]
    
    # if the selected attributes don't have a sufficient number of images try again
    while withattribute.shape[0]<n_images-1:
        attributes_set = np.random.choice(np.arange(40), 2, replace=False)
        withattribute = attributes_train[(attributes_train.iloc[:,attributes_set[0]]==1)&
                    (attributes_train.iloc[:,attributes_set[1]]==1)]
    
        withoutattribute = attributes_train[(attributes_train.iloc[:,attributes_set[0]]!=1)&
                                (attributes_train.iloc[:,attributes_set[1]]!=1)]
    
    # select 9 random images with that attribute
    ixs = np.random.choice(np.arange(withattribute.shape[0]), n_images-1, replace=False)
    images = []
    for ix in ixs:
        image = Image.open('scripts/celebA/img_align_celeba/'+withattribute.iloc[ix, 0])
        image = image.resize((64, 64))
        image = np.asarray(image)
        images.append(image)
        
    # select 1 random image without that attribute
    ix = np.random.choice(np.arange(withoutattribute.shape[0]), 1)
    image = Image.open('scripts/celebA/img_align_celeba/'+str(withoutattribute['Filename'].iloc[ix].values[0]))
    image = image.resize((64, 64))
    image = np.asarray(image)
    images.append(image)
    
    #shuffle the label in the set
    outs = np.zeros((n_images, 1))
    outs[-1] = 1
    ixs = np.arange(n_images)
    np.random.shuffle(ixs)
    images = np.array(images)[ixs]
    outs = outs[ixs]
    attributes = attributes_set
    
    return images, outs, attributes


def create_dataset(data, size=18000, n_images=10):
    images = []
    outs = []
    attributes = []
    for i in tqdm(range(size)):
        im, out, attr = single_set_creation(data, n_images)
        images.append(im)
        outs.append(out)
        attributes.append(attr)
    return np.array(images), np.array(outs), np.array(attributes) 


if __name__ == "__main__":
    
    np.random.seed(0)
    
    # This code assumes that data was downloaded from 
    # https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    # and saved in a folder called celebA
    attributes = pd.read_csv("scripts/celebA/list_attr_celeba.csv")
    identities = pd.read_csv("scripts/celebA/identity_CelebA.txt", sep=' ', header=None)
    attributes.columns = ['Filename']+list(attributes.columns[:-1])
    
    # randomly separating ids from train and test 
    ids = np.unique(identities[1])
    ids = np.random.permutation(ids)
    length = ids.shape[0]
    ids_train = ids[:length//2]
    ids_test = ids[length//2:]
    
    print(f"Number of identities in train is {ids_train.shape[0]}, number of identities in test is {ids_test.shape[0]}")
    
    # get all filenames associated to identities 
    identities_train = identities[identities[1].isin(ids_train)]
    identities_test = identities[identities[1].isin(ids_test)]
    
    # get separate datasets for attributes
    attributes_train = attributes[attributes['Filename'].isin(identities_train[0])]
    attributes_test = attributes[attributes['Filename'].isin(identities_test[0])]
    
    images, outs, attributes = create_dataset(attributes_train)
    
    
    hf = h5py.File('datasets/data/CelebA_train_10.h5', 'w')
    hf.create_dataset('train_data', data=images)
    hf.create_dataset('train_labels', data=outs)
    hf.create_dataset('train_attributes', data=np.array(attributes))
    hf.close()
    
    images, outs, attributes = create_dataset(attributes_test)
    hf = h5py.File('datasets/data/CelebA_test_10.h5', 'w')
    hf.create_dataset('test_data', data=images)
    hf.create_dataset('test_labels', data=outs)
    hf.create_dataset('test_attributes', data=np.array(attributes))
    hf.close()