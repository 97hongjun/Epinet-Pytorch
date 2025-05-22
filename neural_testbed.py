import torch
import torch.nn as nn
from model import MLP
from torch import Tensor
from data import FastTensorDataLoader

def generateSample(input_dim, model, dataset_size, generator=None):
    with torch.no_grad():
        X = torch.normal(torch.zeros((dataset_size, input_dim)), torch.ones((dataset_size, input_dim)), generator=generator)
        # add softmax
        probs = model(X)
        Y = torch.bernoulli(probs[:,0], generator=generator)
    return X, Y, probs

def get_dataloaders(X: Tensor, Y: Tensor, probs: Tensor, batch_size: int, shuffle: bool, keep_last: bool,
                    generator: torch.Generator, device: str):
    return FastTensorDataLoader(X, Y, probs, batch_size=batch_size, shuffle=shuffle,
                                keep_last=keep_last, generator=generator, device=device)

def train(model: nn.Module, train_loader: FastTensorDataLoader, num_epochs: int, optimizer: torch.optim.Optimizer):
    pass

def evaluate(model: nn.Module, test_loader: FastTensorDataLoader):
    pass

