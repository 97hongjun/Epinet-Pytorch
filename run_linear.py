from neural_testbed import *
from model import LinearModel

if __name__ == "__main__":
    input_dim = 64
    swa = True
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    data_seed = 42
    agent_seed = 43
    data_generator = torch.Generator(device="cpu").manual_seed(data_seed)
    data_gpu_generator = torch.Generator(device="cuda").manual_seed(data_seed)
    agent_generator = torch.Generator(device="cpu").manual_seed(agent_seed)
    agent_gpu_generator = torch.Generator(device="cuda").manual_seed(agent_seed)
    model = LinearModel(input_dim, generator=data_gpu_generator).to(device)
    agent = LinearModel(input_dim, generator=agent_gpu_generator).to(device)
    model.apply(model.init_xavier_uniform)
    agent.apply(agent.init_xavier_uniform)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    if swa:
        swa_model = AveragedModel(agent)
        filename = f"logs/training_loss_in{input_dim}_data_seed{data_seed}_agent_seed{agent_seed}_swa.log"
    else:
        filename = f"logs/training_loss_in{input_dim}_data_seed{data_seed}_agent_seed{agent_seed}.log"
    train_samples = 30
    total_loss = 0.0
    total_swa_loss = 0.0
    counter = 0
    for _ in range(2**train_samples//2**22):
        print(f"Training {counter} samples")
        X_train, Y_train, probs_train = generateLogistic(input_dim, model, 2**22, data_generator, device)  
        train_loader = get_dataloaders(X_train, Y_train, probs_train, batch_size=128, shuffle=True, keep_last=True, generator=agent_generator, device=device)
        if swa:
            agent, counter, total_loss, total_swa_loss = train_single_pass_logistic(agent, train_loader, optimizer, device, filename, log_freq=2**10, swa_model=swa_model, counter=counter, total_loss=total_loss, total_swa_loss=total_swa_loss)
        else:
            agent, counter, total_loss, _ = train_single_pass_logistic(agent, train_loader, optimizer, device, filename, log_freq=2**10, counter=counter, total_loss=total_loss)
        del X_train, Y_train, probs_train, train_loader
        torch.cuda.empty_cache()