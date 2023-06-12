import numpy as np
from torch.utils.data import Dataset
import yaml
import torch
import numpy as np
import statistics
from tqdm import tqdm
import os
from packages.networks import model_initializer, optim_initializer


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    parameters = []
    for p in model.parameters():
        if p.requires_grad:
            parameters.append(np.prod(p.size()))
    total_params = np.sum(parameters)
    return total_params

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if val_loss < self.val_loss_min:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss

def training_loop(config, training_loader, validation_loader, tuning_title):
    tuning_folder = os.path.join("packages/tuning", tuning_title)
    # Check to see if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create mask to ensure unidirectional prediction 
    mask = torch.ones(config["block_size"], config["block_size"], requires_grad=False).tril().to(device)
    #Create model, optimizer and loss function
    model = model_initializer(config, mask)
    model = model.to(device)
    optimizer = optim_initializer(config, model)
    loss_func = torch.nn.MSELoss()

    print(f"Total number of paramereters: {count_parameters(model)}")

    experiment_loss = np.zeros((config["num_epochs"], 2))
    # Training loop
    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}:")
        train_loss = []
        val_losses = []
        for sample in tqdm(training_loader, f"Training"):
            x, y = sample["features"], sample["targets"]
            x, y = x.to(device).float(), y.to(device).float()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        
        with torch.no_grad():
            for val_sample in tqdm(validation_loader, f"Validation"):
                x_val, y_val = val_sample["features"], val_sample["targets"]
                x_val, y_val = x_val.to(device).float(), y_val.to(device).float()
                y_val_pred = model(x_val)
                val_loss = loss_func(y_val_pred, y_val)
                val_losses.append(val_loss.item())
            epoch_val_loss = statistics.fmean(val_losses)
            epoch_train_loss = statistics.fmean(train_loss)
            experiment_loss[epoch, :] = epoch_train_loss, epoch_val_loss
        
            print(experiment_loss[epoch, :], "\n")
    
    np.savetxt(os.path.join(tuning_folder, "loss.csv"), experiment_loss, delimiter=',')
    torch.save(model.state_dict(), os.path.join(tuning_folder, f"{tuning_title}.pth"))



