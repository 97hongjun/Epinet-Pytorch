import torch
import torch.nn as nn
from model import MLP
from torch import Tensor
from data import FastTensorDataLoader

def generateSample(input_dim, model, dataset_size, generator=None):
    with torch.no_grad():
        X = torch.normal(torch.zeros((dataset_size, input_dim)), torch.ones((dataset_size, input_dim)), generator=generator)
        # add softmax
        probs = nn.Softmax(dim=-1)(model(X))
        Y = torch.bernoulli(probs[:,1], generator=generator)
    return X, Y, probs

def get_dataloaders(X: Tensor, Y: Tensor, probs: Tensor, batch_size: int, shuffle: bool, keep_last: bool,
                    generator: torch.Generator, device: str):
    return FastTensorDataLoader(X, Y, probs, batch_size=batch_size, shuffle=shuffle,
                                keep_last=keep_last, generator=generator, device=device)

def train(model: nn.Module, train_loader: FastTensorDataLoader, num_epochs: int, optimizer: torch.optim.Optimizer, device: str):
    model.train()
    for _ in range(num_epochs):
        for x, y, probs in train_loader:
            x, y, probs = x.to(device), y.to(device), probs.to(device)
            pred = nn.Softmax(dim=-1)(model(x))

            pred_prob = pred[torch.arange(len(pred)), y]
            true_prob = pred[torch.arange(len(probs)), y]
            kl = torch.log(true_prob) - torch.log(pred_prob)
            loss = torch.mean(kl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
            
def evaluate(model: nn.Module, test_loader: FastTensorDataLoader, device: str):
    model.eval()
    with torch.no_grad():
        total_kl = 0.0
        num_samples = 0
        for x, y, probs in test_loader:
            x, y, probs = x.to(device), y.to(device), probs.to(device)
            pred = nn.Softmax(dim=-1)(model(x))

            pred_prob = pred[torch.arange(len(pred)), y]
            true_prob = pred[torch.arange(len(probs)), y]
            kl = torch.log(true_prob) - torch.log(pred_prob)
            loss = torch.sum(kl)
            total_kl += loss.item()
            num_samples += len(x)
    return total_kl / float(num_samples)

