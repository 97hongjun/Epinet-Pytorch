from neural_testbed import *
from model import LinearModel

if __name__ == "__main__":
    input_dim = 64
    swa = True
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    data_generator = torch.Generator(device="cpu").manual_seed(42)
    data_gpu_generator = torch.Generator(device="cuda").manual_seed(42)
    agent_generator = torch.Generator(device="cpu").manual_seed(43)
    agent_gpu_generator = torch.Generator(device="cuda").manual_seed(43)
    model = LinearModel(input_dim, generator=data_gpu_generator).to(device)
    agent = LinearModel(input_dim, generator=agent_gpu_generator).to(device)
    model.apply(model.init_xavier_uniform)
    agent.apply(agent.init_xavier_uniform)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    if swa:
        swa_model = AveragedModel(agent)
        filename = f"logs/training_loss_in{input_dim}_swa.log"
    else:
        filename = f"logs/training_loss_in{input_dim}.log"
    train_samples = 26
    counter = 0
    for _ in range(2**train_samples//2**23):
        print(f"Training {counter} samples")
        X_train, Y_train, probs_train = generateLogistic(input_dim, model, 2**23, data_generator, device)  
        train_loader = get_dataloaders(X_train, Y_train, probs_train, batch_size=128, shuffle=True, keep_last=True, generator=agent_generator, device=device)
        import pdb; pdb.set_trace()
        if swa:
            agent, counter = train_single_pass_logistic(agent, train_loader, optimizer, device, filename, log_freq=2**10, swa_model=swa_model, counter=counter)
        else:
            agent, counter = train_single_pass_logistic(agent, train_loader, optimizer, device, filename, log_freq=2**10, counter=counter)
        del X_train, Y_train, probs_train
        torch.cuda.empty_cache()