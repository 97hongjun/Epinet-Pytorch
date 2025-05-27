import torch
import torch.nn as nn
from model import MLP
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
            
def train_single_pass(model: nn.Module, train_loader: FastTensorDataLoader, optimizer: torch.optim.Optimizer, 
device: str, filename: str, log_freq: int, swa_model: Optional[AveragedModel]=None, swa_scheduler: Optional[SWALR]=None) -> nn.Module:
    model.train()
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    total_samples = 0
    total_loss = 0.0
    total_swa_loss = 0.0
    for i, (x, y, probs) in enumerate(train_loader):
        x, y, probs = x.to(device), y.to(device), probs.to(device)
        pred = nn.Softmax(dim=-1)(model(x))

        pred_prob = pred[torch.arange(len(pred)), y]
        true_prob = probs[torch.arange(len(probs)), y]
        kl = torch.log(true_prob) - torch.log(pred_prob)
        loss = torch.sum(kl)
        total_samples += len(x)
        total_loss += loss.item()

        if swa_model is not None:
            swa_pred = nn.Softmax(dim=-1)(swa_model(x))
            swa_pred_prob = swa_pred[torch.arange(len(swa_pred)), y]
            swa_true_prob = probs[torch.arange(len(probs)), y]
            swa_kl = torch.log(swa_true_prob) - torch.log(swa_pred_prob)
            swa_loss = torch.sum(swa_kl)
            total_swa_loss += swa_loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if swa_model is not None and swa_scheduler is not None:
            swa_model.update_parameters(model)
            # swa_scheduler.step()
        if i % log_freq == 0:
            if i == 0:
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
    return model

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
    input_dim = 10
    hidden_dims_model = [64, 64]
    output_dims = 2
    hidden_dims_agent = [256, 256]
    swa = True
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    data_generator = torch.Generator(device="cpu").manual_seed(42)
    data_gpu_generator = torch.Generator(device="cuda").manual_seed(42)
    agent_generator = torch.Generator(device="cpu").manual_seed(43)
    agent_gpu_generator = torch.Generator(device="cuda").manual_seed(43)
    model = MLP(input_dim, hidden_dims_model, output_dims, generator=data_gpu_generator).to(device)
    agent = MLP(input_dim, hidden_dims_agent, output_dims, generator=agent_gpu_generator).to(device)
    model.apply(model.init_xavier_uniform)
    agent.apply(agent.init_xavier_uniform)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    if swa:
        swa_model = AveragedModel(agent)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        filename = f"logs/training_loss_in{input_dim}_hid{hidden_dims_model[0]}_out{output_dims}_hid{hidden_dims_agent[0]}_swa.log"
    else:
        filename = f"logs/training_loss_in{input_dim}_hid{hidden_dims_model[0]}_out{output_dims}_hid{hidden_dims_agent[0]}.log"
    X_train, Y_train, probs_train = generateSample(input_dim, model, 2**26, data_generator, device)  
    train_loader = get_dataloaders(X_train, Y_train, probs_train, batch_size=128, shuffle=True, keep_last=True, generator=agent_generator, device=device)
    # X_test, Y_test, probs_test = generateSample(input_dim, model, 2**16, data_generator, device)
    # test_loader = get_dataloaders(X_test, Y_test, probs_test, batch_size=128, shuffle=False, keep_last=True, generator=agent_generator, device=device)
    # agent = train(agent, train_loader, test_loader, num_epochs=100, optimizer=optimizer, device=device, filename=filename)
    if swa:
        agent = train_single_pass(agent, train_loader, optimizer, device, filename, log_freq=2**11, swa_model=swa_model, swa_scheduler=swa_scheduler)
    else:
        agent = train_single_pass(agent, train_loader, optimizer, device, filename, log_freq=2**11)
    # kl = evaluate(agent, test_loader, device=device)
    # print(f"KL: {kl}")