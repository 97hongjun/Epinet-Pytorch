import torch
import torch.nn as nn
from model import MLP
from torch import Tensor
from data import FastTensorDataLoader
import os

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

def train(model: nn.Module, train_loader: FastTensorDataLoader, test_loader: FastTensorDataLoader, num_epochs: int, optimizer: torch.optim.Optimizer, device: str) -> nn.Module:
    model.train()
    # Initialize loss logging
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_loss.log')
    
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
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{avg_train_loss},{test_loss}\n")
            
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}")
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
    hidden_dims = [50, 50]
    output_dims = 2
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = MLP(input_dim, hidden_dims, output_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data_generator = torch.Generator(device="cpu").manual_seed(42)
    agent_generator = torch.Generator(device="cpu").manual_seed(42)
    X_train, Y_train, probs_train = generateSample(input_dim, model, 2**16, data_generator, device)  
    train_loader = get_dataloaders(X_train, Y_train, probs_train, batch_size=128, shuffle=True, keep_last=True, generator=agent_generator, device=device)
    X_test, Y_test, probs_test = generateSample(input_dim, model, 2**16, data_generator, device)
    test_loader = get_dataloaders(X_test, Y_test, probs_test, batch_size=128, shuffle=False, keep_last=True, generator=agent_generator, device=device)
    model = train(model, train_loader, test_loader, num_epochs=100, optimizer=optimizer, device=device)
    kl = evaluate(model, test_loader, device=device)
    print(f"KL: {kl}")