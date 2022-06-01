import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from models import DeepSetsSum
from datasets import NormalVarOODDataset, NormalVarDataset
from train import train
from experiments.utils import count_parameters


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Feature Norm Analysis")
    parser.add_argument("--norm", default="set_norm", type=str,
                        help="none|layer_norm|feature_norm|set_norm")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--task", default="ood", type=str, 
                        help="ood|indistr")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--turnoff_wandb', action="store_true")
    
    args = parser.parse_args()

    if not args.turnoff_wandb:
        import wandb
        wandb.init(project='featurenorm_analysis')
     
   
    _train = NormalVarOODDataset(N=10000, n_samples=1000, 
                                n_dim=100, test=False, ood = args.task=='ood')
    _test = NormalVarOODDataset(N=1000, n_samples=1000, 
                                n_dim=100, test=True, ood = args.task=='ood') 
    
    g = torch.Generator()
    g.manual_seed(0)
    train_generator = DataLoader(_train,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=True)
    test_generator = DataLoader(_test,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)
    
    print("Loaded dataset..")
    torch.manual_seed(args.seed)
    model = DeepSetsSum(n_inputs=100,
                n_outputs=1,
                n_enc_layers=25,
                norm=args.norm, sample_size=1000)
    
    num_params = count_parameters(model)
    print(f"Loaded model, number of parameters: {num_params}")
    if not args.turnoff_wandb:
        wandb.run.summary["param_count"] = num_params
    print(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    model = train(model, optimizer, 
                   train_generator, test_generator, 
                   args.task,
                   n_epochs= 30*args.batch_size//2,  
                   n_outputs=1,
                   use_wandb=not args.turnoff_wandb, 
                   accum_steps=1)

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/fn_comparison", exist_ok=True)

    filename =  f"results/fn_comparison/{args.model}_{args.task}_{args.norm}_{args.seed}.pt"
    torch.save(model.state_dict(), filename)