import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from models import SetTransformer, SetTransformer2, SetTransformer2Agg
from train import train
from experiments.utils import count_parameters, get_dataset, N_INPUTS, N_OUTPUTS, seed_worker


def get_model(name, task, norm, residual_pipeline, size):
    sample_size = 1000 if 'mnistvar' not in task else 10
    if name == 'settransformer2':
        if residual_pipeline == "he":
            model = SetTransformer2(n_inputs=N_INPUTS[task],
                        n_outputs=N_OUTPUTS[task],
                        n_enc_layers=size, 
                        norm=norm, sample_size=sample_size)
        elif residual_pipeline == "agg":
            model = SetTransformer2Agg(n_inputs=N_INPUTS[task],
                        n_outputs=N_OUTPUTS[task],
                        n_enc_layers=size,
                        norm=norm, sample_size=sample_size)
        else:
            raise ValueError(f'Unknown residual_pipeline {residual_pipeline}')
    elif name == 'settransformer':
        model = SetTransformer(n_inputs=N_INPUTS[task],
                      n_outputs=N_OUTPUTS[task],
                      n_enc_layers=size,
                      norm=norm, sample_size=sample_size) 
    else:
        raise ValueError(f'Unknown model {name}')
    return model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Set Transformer Analysis")
    parser.add_argument("--task", default="hematocrit", type=str, 
                        help="hematocrit|pointcloud|mnistvar|normalvar")
    parser.add_argument("--model", default="settransformer2", type=str, 
                        help="settransformer|settransformer2")
    parser.add_argument("--norm", default="set_norm", type=str,
                        help="none|layer_norm|feature_norm|set_norm")
    parser.add_argument("--residual_pipeline", default="he", type=str,
                        help="he|agg")  # ignored for model == settransformer
    parser.add_argument("--size", default=16, type=int,
                        help="numer_of_layers")
    parser.add_argument("--store_gradient", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--turnoff_wandb', action="store_true")
    parser.add_argument("--accum_steps", default=1, type=int)
    
    args = parser.parse_args()

    if not args.turnoff_wandb:
        import wandb
        wandb.init(project='st_debugging')
        
    _train, _test = get_dataset(args.task)

    g = torch.Generator()
    g.manual_seed(0)

    train_generator = DataLoader(_train,
                                batch_size=64//args.accum_steps,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=True,
                                worker_init_fn=seed_worker,
                                generator=g)
    test_generator = DataLoader(_test,
                            batch_size=64//args.accum_steps,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True,
                            worker_init_fn=seed_worker,
                            generator=g)
    
    print("Loaded dataset..")
    
    torch.manual_seed(args.seed)
    model = get_model(args.model, args.task, args.norm, args.residual_pipeline, args.size)
    
    num_params = count_parameters(model)
    print(f"Loaded model, number of parameters: {num_params}")
    if not args.turnoff_wandb:
        wandb.run.summary["param_count"] = num_params
    print(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    model = train(model, optimizer, 
                   train_generator, test_generator, 
                   args.task,
                   n_epochs= 30 if args.task == 'hematocrit' else 50, 
                   n_outputs=N_OUTPUTS[args.task],
                   use_wandb=not args.turnoff_wandb, 
                   accum_steps=args.accum_steps, store_gradient=args.store_gradient==1)

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/settransformer", exist_ok=True)

    filename =  f"results/settransformer/{args.task}_{args.model}_{args.norm}_{args.seed}.pt"
    torch.save(model.state_dict(), filename)