import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from models import DeepSets2Conv,  DeepSetsConv, SetTransformerConv, SetTransformer2Conv
from datasets import CelebAAnomalyDetection
from train import train
from experiments.utils import count_parameters, seed_worker


def get_model(name, norm, residual_pipeline, size):
    sample_size = 10
    if name == 'deepsets2':
        model = DeepSets2Conv(n_outputs=10,
                      n_enc_layers=size,
                      norm=norm, sample_size=sample_size,
                      res_pipeline=residual_pipeline)
    elif name == 'deepsets':
        model = DeepSetsConv(n_outputs=10,
                      n_enc_layers=size*2 + 1, #doubling number of layers to match
                      norm=norm, sample_size=sample_size)
    elif name == 'settransformer':
        model = SetTransformerConv(n_outputs=10,
                      n_enc_layers=size, 
                      norm=norm, sample_size=sample_size)
    elif name == 'settransformer2':
        model = SetTransformer2Conv(n_outputs=10,
                      n_enc_layers=size,
                      norm=norm, sample_size=sample_size)
    else:
        raise ValueError(f'Unknown model {name}')
    return model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DeepSet Analysis")
    parser.add_argument("--model", default="deepsets2", type=str, 
                        help="deepsets|deepsets2")
    parser.add_argument("--norm", default="set_norm", type=str,
                        help="none|layer_norm|feature_norm|set_norm")
    parser.add_argument("--residual_pipeline", default="he", type=str,
                        help="resnet|he") # gets ignored in case model == deepsets
    parser.add_argument("--size", default=25, type=int,
                        help="number_of_layers")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--turnoff_wandb', action="store_true")
    parser.add_argument("--accum_steps", default=1, type=int)
    
    args = parser.parse_args()

    if not args.turnoff_wandb:
        import wandb
        wandb.init(project='deepsets_analysis')
        
    _train = CelebAAnomalyDetection(test=False)
    _test = CelebAAnomalyDetection(test=True)
    
    g = torch.Generator()
    g.manual_seed(0)

    train_generator = DataLoader(_train,
                                batch_size=64//2,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=True,
                                worker_init_fn=seed_worker,
                                generator=g)
    test_generator = DataLoader(_test,
                            batch_size=64//2,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True,
                            worker_init_fn=seed_worker,
                            generator=g)
    
    print("Loaded dataset..")
    
    torch.manual_seed(args.seed)
    model = get_model(args.model, args.norm,
                      args.residual_pipeline, args.size)
    
    num_params = count_parameters(model)
    print(f"Loaded model, number of parameters: {num_params}")
    if not args.turnoff_wandb:
        wandb.run.summary["param_count"] = num_params
    print(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    model = train(model, optimizer, 
                   train_generator, test_generator, 
                   'setanomaly',
                   n_epochs= 50, 
                   n_outputs=10,
                   use_wandb=not args.turnoff_wandb, 
                   accum_steps=args.accum_steps)

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/setanomaly", exist_ok=True)

    if model == 'deepsets2':
        filename =  f"results/setanomaly/{args.norm}_{args.residual_pipeline}_{args.seed}.pt"
    else:
        filename =  f"results/setanomaly/{args.residual_pipeline}_{args.norm}_{args.seed}.pt"
    torch.save(model.state_dict(), filename)