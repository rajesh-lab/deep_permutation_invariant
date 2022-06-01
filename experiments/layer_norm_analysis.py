import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from models import DeepSetsSum, DeepSetsMax, SetTransformer
from datasets import WeightedAvgDataset
from train import train
from experiments.utils import count_parameters


def get_model(model, task, norm):
    if model == 'deepsets':
        model = DeepSetsMax if task == 'pointcloud' else DeepSetsSum
        model = model(n_inputs=100,
                n_outputs=1,
                n_enc_layers=3,
                norm=norm, sample_size=1000)
    else:
        model = SetTransformer(n_inputs=100,
                n_outputs=1,
                n_enc_layers=2,
                norm=norm, sample_size=1000)
    return model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Layer Norm Analysis")
    parser.add_argument("--task", default="normal", type=str, 
                        help="normal|categorical")
    parser.add_argument("--norm", default="none", type=str,
                        help="none|layer_norm")
    parser.add_argument("--model", default="settransformer", type=str, 
                        help="settransformer|deepsets")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--turnoff_wandb', action="store_true")
    
    args = parser.parse_args()

    if not args.turnoff_wandb:
        import wandb
        wandb.init(project='deepsets_analysis')

        
    _train = WeightedAvgDataset(N=10000, n_samples=1000, 
                                n_dim=100, data_type=args.task)
    _test = WeightedAvgDataset(N=1000, n_samples=1000, 
                                n_dim=100, data_type=args.task)                           
    train_generator = DataLoader(_train,
                                batch_size=32,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=True)
    test_generator = DataLoader(_test,
                            batch_size=32,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)
    
    print("Loaded dataset..")
    
    model  = get_model(args.model, args.task, args.norm)
    
    num_params = count_parameters(model)
    print(f"Loaded model, number of parameters: {num_params}")
    if not args.turnoff_wandb:
        wandb.run.summary["param_count"] = num_params
    print(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    model = train(model, optimizer, 
                   train_generator, test_generator, 
                   args.task,
                   n_epochs= 30, 
                   n_outputs=1,
                   use_wandb=not args.turnoff_wandb, 
                   seed=args.seed,
                   accum_steps=2)

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/ln_comparison", exist_ok=True)

    filename =  f"results/ln_comparison/{args.model}_{args.task}_{args.norm}_{args.seed}.pt"
    torch.save(model.state_dict(), filename)