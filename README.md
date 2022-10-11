# DeepSets++ and SetTransformer++

Code for ICML 2022 paper "Set Norm and Equivariant Skip Connections: Putting the Deep in Deep Sets."

Lily Zhang, Veronica Tozzo, John M. Higgins, Rajesh Ranganath 


# Getting started

Clone the repository on your computer with the following command 

```https://github.com/veronicatozzo/deep_permutation_invariant.git```


### Requirements 
The code was tested on Linux Os with Python 3.7.9, CUDA version 10.1, GCC version 6.2. 

### Install libraries 
The required libraries are listed in `requirements.txt`, we suggest you create a virtual environment with [Anaconda](https://www.anaconda.com) and install the libraries in it. 

```
cd deep_permutation_invariant 
conda create --name deepperminv python=3.7.9
conda activate deepperminv
pip install -r requirements.txt
```

### Download datasets
We provide a novel single-cell benchmark datasets for prediction sets, called Flow-RBC. The dataset consists of 98,240 train and 23,104 test red blood cell (RBC) distributions. Each distribution consists of volume and hemoglobin mass flow cytometry measurements collected retrospectively at Massachussets General Hospital under an existing IRB-approved research protocol. 
The regression task consists in predicting, from a distribution, the hematocrit level measured on the same speciment. Hematocrit is the fraction of blood volume occupied by red blood cells and good prediction outcomes suggest a stronger relationship between single cells properties and aggregated population properties in the human blood than previously known.

Flow-RBC can be downloaded at the [this link](https://cims.nyu.edu/~lhz209/flowrbc).

All other set datasets (with the exception of Anemia which is not publicly available), can be derived by the following open-source datasets:
1. Point Cloud: [ModelNet40](https://modelnet.cs.princeton.edu/). We use the HDF5 files downloaded from the [Pointnet repository](https://github.com/charlesq34/pointnet) from [this link](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip). (As of Oct 2022, the issued certificate has expired, so downloading will require an additional `--no-check-certificate` flag.)
2. MNIST Variance: [MNIST](https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz).
3. CelebA Set Anomaly: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

See `scripts` for the scripts to generate the h5 files of the datasets used in this repository.


# Code
The code is organized as follows:
- `deep_permutation_invariant/models`: re-implementations of SetTransformer [Lee et al. 2019](http://proceedings.mlr.press/v97/lee19d/lee19d.pdf) and DeepSets [Zaheer et al. 2017](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) for any depth of the encoder and for any lengths of the input sets. It also contains the implementation of SetTransformer++ and DeepSets++.

- `deep_permutation_invariant/datasets`: dataloaders for the real and synthetic datasets used in the paper. The downloaded datasets will be stored in a subfolder `data`

- `deep_permutation_invariant/experiments`: experiment scripts

- `deep_permutation_invariant/configs`: hyperparameters setup for the experiments

# Running experiments 
The scripts for the main results in the papers are `experiments/deep_sets_analysis.py` and `experiments/set_transformer_analysis.py`. Each script takes in input a model (`deepsets` or `deepsets2`, `settransformer` or `settransformer2`), a task (`hematocrit`, `pointcloud`, `normalvar`, `mnistvar`), and a norm (`set_norm`, `layer_norm`, `feature_norm`).

To run an experiment:

```
python experiments/set_transformer_analysis.py --model=settransformer2 --norm=set_norm  --task=hematocrit --seed=0 --turnoff_wandb
```

the script will automatically save the trained model in the folder `results/settransformer` (`results/deepsets`).

### Memory requirements
The code requires 24GB GPUs to run with a batch size of 64. If using GPUs that do not meet this memory requirement, one can use gradient accumulation by specifying `accum_steps` to a value greater than 1.


### Running with wandb 
To log all the results to [Weights & Biases (wandb)](https://wandb.ai/home), after creating an account and installing wandb in your conda environment 

```
conda install -c conda-forge wandb
```

login with your api-key (you'll find it in the settings on your wandb account).

```
wandb login <api-key>
```

You can then create the sweeps using the provided configuration files 

``` 
wandb sweep configs/set_transformer.yaml
```

you will see in the prompt the following lines 


```
wandb: Creating sweep from: configs/settransformer.yaml
wandb: Created sweep with ID: <sweepid>
wandb: View sweep at: https://wandb.ai/<username>/results/sweeps/<sweepid>
wandb: Run sweep agent with: wandb agent <username>/results/<sweepid>

```

You can kick off the runs executing `wandb agent <username>/results/<sweepid>` directly or with any scheduler. All the train and test loss curves will be logged at `https://wandb.ai/<username>/results/sweeps/<sweepid>`. 

