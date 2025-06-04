import torch
import torch.nn as nn
from model import MLP, Epinet
from torch import Tensor
from data import FastTensorDataLoader
import os
from torch.optim.swa_utils import AveragedModel, SWALR
from typing import Optional

def generateSample(input_dim, model, dataset_size, generator=None, device="cpu"):
    with torch.no_grad():
        X = torch.normal(torch.zeros((dataset_size, input_dim)), torch.ones((dataset_size, input_dim)), generator=generator).to(device)
        # add softmax
        probs = nn.Softmax(dim=-1)(model(X)).to("cpu")
        Y = torch.bernoulli(probs[:,1], generator=generator).int()
    return X.to("cpu"), Y, probs

def generateLogistic(input_dim, model, dataset_size, generator=None, device="cpu"):
    with torch.no_grad():
        X = torch.normal(torch.zeros((dataset_size, input_dim)), torch.ones((dataset_size, input_dim)), generator=generator).to(device)
        # add softmax
        probs = nn.Sigmoid()(model(X)).to("cpu")
        probs = torch.cat((1-probs, probs), dim=-1)
        Y = torch.bernoulli(probs[:,1], generator=generator).int()
    return X.to("cpu"), Y, probs

def get_dataloaders(X: Tensor, Y: Tensor, probs: Tensor, batch_size: int, shuffle: bool, keep_last: bool,
                    generator: torch.Generator, device: str):
    return FastTensorDataLoader(X, Y, probs, batch_size=batch_size, shuffle=shuffle,
                                keep_last=keep_last, generator=generator, device=device)

def train(model: nn.Module, train_loader: FastTensorDataLoader, test_loader: FastTensorDataLoader, num_epochs: int, optimizer: torch.optim.Optimizer, device: str, filename: str) -> nn.Module:
    model.train()
    # Initialize loss logging
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for x, y, probs in train_loader:
            x, y, probs = x.to(device), y.to(device), probs.to(device)
            pred = nn.Softmax(dim=-1)(model(x))

            pred_prob = pred[torch.arange(len(pred)), y]
            true_prob = probs[torch.arange(len(probs)), y]
            kl = torch.log(true_prob) - torch.log(pred_prob)
            loss = torch.mean(kl)
            
            # Accumulate loss for this epoch
            epoch_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate and log both training and test loss
        avg_train_loss = epoch_loss / num_batches
        test_loss = evaluate(model, test_loader, device)
        
        # Log both losses to file
        if epoch == 0:
            with open(log_file, 'w') as f:
                f.write("epoch,train_loss,test_loss\n")
        else:
            with open(log_file, 'a') as f:
                f.write(f"{epoch},{avg_train_loss},{test_loss}\n")
            
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}")
    return model
            
def train_single_pass_logistic(model: nn.Module, train_loader: FastTensorDataLoader, optimizer: torch.optim.Optimizer,
                               device: str, filename: str, log_freq: int, swa_model: Optional[AveragedModel]=None, counter: int=0, total_loss: float=0.0, total_swa_loss: float=0.0) -> tuple[nn.Module, int, float, float]:
    model.train()
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    total_samples = counter
    total_loss = total_loss
    total_swa_loss = total_swa_loss
    for i, (x, y, probs) in enumerate(train_loader):
        x, y, probs = x.to(device), y.to(device), probs.to(device)
        pred = nn.Sigmoid()(model(x))
        pred = torch.cat((1-pred, pred), dim=-1)

        pred_prob = pred[torch.arange(len(pred)), y]
        true_prob = probs[torch.arange(len(probs)), y]
        loss = torch.sum(torch.log(true_prob) - torch.log(pred_prob))
        total_samples += len(x)

        true_kl = torch.log(true_prob) - torch.log(pred_prob)
        true_kl2 = torch.log(1.0 - true_prob) - torch.log(1.0 - pred_prob)
        true_kl_full = true_prob * true_kl + (1.0 - true_prob) * true_kl2
        total_loss += torch.sum(true_kl_full).item()

        if swa_model is not None:
            swa_pred = nn.Sigmoid()(swa_model(x))
            swa_pred = torch.cat((1-swa_pred, swa_pred), dim=-1)
            swa_pred_prob = swa_pred[torch.arange(len(swa_pred)), y]
            swa_true_prob = probs[torch.arange(len(probs)), y]
            swa_kl = torch.log(swa_true_prob) - torch.log(swa_pred_prob)
            swa_kl2 = torch.log(1.0 - swa_true_prob) - torch.log(1.0 - swa_pred_prob)
            swa_kl_full = swa_true_prob * swa_kl + (1.0 - swa_true_prob) * swa_kl2
            swa_loss = torch.sum(swa_kl_full)
            total_swa_loss += swa_loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if swa_model is not None:
            swa_model.update_parameters(model)
            # swa_scheduler.step()
        if i % log_freq == 0:
            if total_samples == len(x):
                with open(log_file, 'w') as f:
                    if swa_model is not None:   
                        f.write("batch,loss,swa_loss\n")
                    else:
                        f.write("batch,loss\n")
            else:
                with open(log_file, 'a') as f:
                    if swa_model is not None:
                        f.write(f"{total_samples},{total_loss/total_samples},{total_swa_loss/total_samples}\n")
                    else:
                        f.write(f"{total_samples},{total_loss/total_samples}\n")
    return model, total_samples, total_loss, total_swa_loss



def train_single_pass(model: nn.Module, train_loader: FastTensorDataLoader, optimizer: torch.optim.Optimizer, 
                      device: str, filename: str, log_freq: int, swa_model: Optional[AveragedModel]=None, counter: int=0,
                      total_loss: float=0.0, total_swa_loss: float=0.0, lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None) -> tuple[nn.Module, int, float, float]:
    model.train()
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    total_samples = counter
    total_loss = total_loss
    total_swa_loss = total_swa_loss
    for i, (x, y, probs) in enumerate(train_loader):
        x, y, probs = x.to(device), y.to(device), probs.to(device)
        pred = nn.Softmax(dim=-1)(model(x))

        pred_prob = pred[torch.arange(len(pred)), y]
        true_prob = probs[torch.arange(len(probs)), y]
        kl = torch.log(true_prob) - torch.log(pred_prob)

        loss = torch.sum(kl)
        total_samples += len(x)

        true_kl = torch.log(probs) - torch.log(pred)
        true_kl_full = probs * true_kl
        total_loss += torch.sum(true_kl_full).item()
        
        if swa_model is not None:
            swa_pred = nn.Softmax(dim=-1)(swa_model(x))
            true_kl_swa = torch.log(probs) - torch.log(swa_pred)
            true_kl_full_swa = probs * true_kl_swa
            total_swa_loss += torch.sum(true_kl_full_swa).item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if swa_model is not None:
            swa_model.update_parameters(model)
        if i % log_freq == 0:
            if total_samples == len(x):
                with open(log_file, 'w') as f:
                    if swa_model is not None:   
                        f.write("batch,loss,swa_loss\n")
                    else:
                        f.write("batch,loss\n")
            else:
                with open(log_file, 'a') as f:
                    if swa_model is not None:
                        f.write(f"{total_samples},{total_loss/total_samples},{total_swa_loss/total_samples}\n")
                    else:
                        f.write(f"{total_samples},{total_loss/total_samples}\n")
    return model, total_samples, total_loss, total_swa_loss

def evaluate_joint_logloss(model: Epinet, x: torch.Tensor, y: torch.Tensor, probs: torch.Tensor, num_indices: int) -> float:
    model.eval()
    with torch.no_grad():
        total_kl = 0.0
        num_samples = 0
        z = torch.randn(num_indices, model.index_dim, generator=model.generator)
        pred = nn.Softmax(dim=-1)(model(x, z))
        # shape of (batch_size * num_indices, output_dim)
        pred = pred.view(batch_size, num_indices, output_dim)
        # shape of (batch_size, num_indices,)
        pred_terms = pred[torch.arange(len(pred)),:, y]

        # shape of (num_indices,)
        joint_preds = torch.product(pred_terms, dim=0)
        joint_pred = torch.mean(joint_preds, dim=0)

        # of shape (output_dim)
        prob_terms = probs[torch.arange(len(probs)),y]
        kl_term = torch.sum(torch.log(prob_terms), dim=0) - torch.log(joint_pred)

        total_kl += kl_term.item()
        num_samples += len(x)
    return total_kl / float(num_samples)

def evaluate(model: nn.Module, test_loader: FastTensorDataLoader, device: str) -> float:
    model.eval()
    with torch.no_grad():
        total_kl = 0.0
        num_samples = 0
        for x, y, probs in test_loader:
            x, y, probs = x.to(device), y.to(device), probs.to(device)
            pred = nn.Softmax(dim=-1)(model(x))

            pred_prob = pred[torch.arange(len(pred)), y]
            true_prob = probs[torch.arange(len(probs)), y]
            kl = torch.log(true_prob) - torch.log(pred_prob)
            loss = torch.sum(kl)
            total_kl += loss.item()
            num_samples += len(x)
    return total_kl / float(num_samples)

if __name__ == "__main__":
    input_dim = 64
    hidden_dims_model = [128, 128]
    output_dims = 2
    hidden_dims_agent = [2048, 2048]
    swa = True
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    optim = 'Adam'
    data_seed = 50
    agent_seed = 51
    data_generator = torch.Generator(device="cpu").manual_seed(data_seed)
    data_gpu_generator = torch.Generator(device="cuda").manual_seed(data_seed)
    agent_generator = torch.Generator(device="cpu").manual_seed(agent_seed)
    agent_gpu_generator = torch.Generator(device="cuda").manual_seed(agent_seed)
    model = MLP(input_dim, hidden_dims_model, output_dims, generator=data_gpu_generator).to(device)
    agent = MLP(input_dim, hidden_dims_agent, output_dims, generator=agent_gpu_generator).to(device)
    model.apply(model.init_xavier_uniform)
    agent.apply(agent.init_xavier_uniform)
    if optim == 'SGD':
        optimizer = torch.optim.SGD(agent.parameters(), lr=0.001)
        lambda1 = lambda x: 1.0/(x + 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    if swa:
        swa_model = AveragedModel(agent)
        if optim == 'SGD':
            filename = f"logs/sgd/training_loss_in{input_dim}_hid{hidden_dims_model[0]}_out{output_dims}_hid{hidden_dims_agent[0]}_data_seed{data_seed}_agent_seed{agent_seed}_swa.log"
        else:
            filename = f"logs/adam/training_loss_in{input_dim}_hid{hidden_dims_model[0]}_out{output_dims}_hid{hidden_dims_agent[0]}_data_seed{data_seed}_agent_seed{agent_seed}_swa.log"
    else:
        if optim == 'SGD':
            filename = f"logs/sgd/training_loss_in{input_dim}_hid{hidden_dims_model[0]}_out{output_dims}_hid{hidden_dims_agent[0]}_data_seed{data_seed}_agent_seed{agent_seed}.log"
        else:
            filename = f"logs/adam/training_loss_in{input_dim}_hid{hidden_dims_model[0]}_out{output_dims}_hid{hidden_dims_agent[0]}_data_seed{data_seed}_agent_seed{agent_seed}.log"
    train_samples = 30
    total_loss = 0.0
    total_swa_loss = 0.0
    counter = 0
    for _ in range(2**train_samples//2**16):
        print(f"Training {counter} samples")
        X_train, Y_train, probs_train = generateSample(input_dim, model, 2**16, data_generator, device)  
        train_loader = get_dataloaders(X_train, Y_train, probs_train, batch_size=128, shuffle=True, keep_last=True, generator=agent_generator, device=device)
        if swa:
            agent, counter, total_loss, total_swa_loss = train_single_pass(agent, train_loader, optimizer, device, filename, log_freq=2**10, swa_model=swa_model, counter=counter, total_loss=total_loss, total_swa_loss=total_swa_loss)
        else:
            agent, counter, total_loss, _ = train_single_pass(agent, train_loader, optimizer, device, filename, log_freq=2**10, counter=counter, total_loss=total_loss)
        del X_train, Y_train, probs_train, train_loader
        torch.cuda.empty_cache()