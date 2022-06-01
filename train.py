import torch
import torch.nn as nn
import numpy as np


def store_gradients(model, step, use_wandb=True):
    import wandb
    l = [module for module in list(model.named_modules())[1:] if isinstance(module[1], nn.Linear)]
    for name, layer in l:
        weight_grad = torch.linalg.norm(layer.weight.grad)
        weight = torch.linalg.norm(layer.weight)
        # bias_grad = torch.linalg.norm(layer.bias.grad)
        if use_wandb:
            wandb.log({f"{name} weights grad": weight_grad.item()}, step=step)
            wandb.log({f"{name} weights": weight.item()}, step=step)
            # wandb.log({f"{name} bias": bias_grad.item()}, step=step)
        

def train(model, optimizer, train_generator, test_generator, task, 
          n_epochs=30, n_outputs=1, use_wandb=False,  accum_steps=1, store_gradient=False):
    
    if use_wandb:
        import wandb
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if 'pointcloud' in task or task == 'setanomaly' or task == 'anemia':
        criterion = nn.CrossEntropyLoss()
        classification = True
    else:
        criterion = nn.MSELoss(reduction="none") 
        classification = False 

    # plot_freq = 20
    plot_freq = 20

    step = 0
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    optimizer.zero_grad()
    num_correct = 0
    for epoch in range(n_epochs):
        train_loss_ = []
        for x, y, lengths in train_generator:
            #print(x.shape, y.shape, lengths.shape)
            x, y, lengths = x.type(dtype).to(device), y.type(dtype).to(device), lengths.to(device)
            if task == "pointcloud_categorical":
                x = x.type(torch.cuda.LongTensor)
            preds = model(x, lengths)
            preds = preds.reshape(x.shape[0], n_outputs)
            if classification:
                y = y.type(torch.cuda.LongTensor) if torch.cuda.is_available() else y.type(torch.LongTensor)
                y = y.squeeze()
                if task == 'setanomaly':
                    y = y.argmax(axis=1)
                    num_correct += (torch.sum(preds.argmax(axis=1) == y).detach().cpu().numpy())/x.shape[0]
                else:
                    num_correct += (torch.sum(preds.argmax(axis=1) == y).detach().cpu().numpy())/x.shape[0]
            else:
                assert preds.shape == y.shape, "{} {}".format(preds.shape, y.shape)
            loss_elements = criterion(preds, y)
            loss = loss_elements.mean()
            train_loss = loss.detach().cpu().numpy()
            
            if np.isnan(loss.detach().cpu().numpy()):
                raise ValueError("Train loss is nan: ", loss)
            
            step += 1
            loss.backward()

            if step % accum_steps == 0:
                optimizer.step()
                if (step % (plot_freq * accum_steps) == 0) and store_gradient:
                    store_gradients(model, step, use_wandb=use_wandb)
                optimizer.zero_grad()

            if step % (plot_freq * accum_steps) == 0:
                if classification:
                    train_acc = num_correct/(plot_freq * accum_steps)
                    num_correct = 0
                test_aux = []
                for x, y, lengths in test_generator:
                    x, y, lengths = x.type(dtype).to(device), y.type(dtype).to(device), lengths.to(device)
                    if task == "pointcloud_categorical":
                        x = x.type(torch.cuda.LongTensor)
                    preds = model(x, lengths)
                    preds = preds.reshape(x.shape[0], n_outputs)
                    if classification:
                        y = y.type(torch.cuda.LongTensor) if torch.cuda.is_available() else y.type(torch.LongTensor)
                        y = y.squeeze()
                        if task == 'setanomaly':
                            y = y.argmax(axis=1)
                            num_correct += (torch.sum(preds.argmax(axis=1) == y).detach().cpu().numpy())/x.shape[0]
                        else:
                            num_correct += torch.sum(preds.argmax(axis=1) == y).detach().cpu().numpy()/x.shape[0]
                    else:
                        assert preds.shape == y.shape, "{} {}".format(preds.shape, y.shape)

                    loss_elements = criterion(preds, y)
                    loss = loss_elements.mean()
                    test_aux.append(loss.detach().cpu().numpy())

                test_loss = np.mean(test_aux)
                if classification:
                    test_acc = num_correct/len(test_generator)
                    num_correct = 0

                if use_wandb:
                    wandb.log({f"{task} test loss": test_loss}, step=step//accum_steps)
                    wandb.log({f"{task} train loss": train_loss}, step=step//accum_steps)
                    if classification:
                        wandb.log({f"{task} train accuracy": train_acc}, step=step//accum_steps)
                        wandb.log({f"{task} test accuracy": test_acc}, step=step//accum_steps)
                if classification:
                    print(f"Epoch: {epoch}, step: {step//accum_steps}, train loss: {train_loss},  "+
                            f"train accuracy: {train_acc}, "+
                            f"test loss: {test_loss},  test acc: {test_acc}")
                else:
                    print(f"Epoch: {epoch}, step: {step//accum_steps}, train loss: {train_loss}, test loss: {test_loss}")
       
    return model