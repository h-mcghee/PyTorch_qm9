import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models.gnn import GNN

def load_data(data_dir = "data/qm9", data_size = 30000, train_fraction = 0.8, val_fraction = 0.1, test_fraction = 0.1, batch_size = 64, target_index = 2):
    qm9 = QM9(root=data_dir)
    y_target = pd.DataFrame(qm9.data.y.numpy())  #so to inspect y targets visually
    qm9.data.y = torch.Tensor(y_target[target_index])

    qm9 = qm9.shuffle()

    train_index = int(data_size * train_fraction)
    test_index = train_index + int(data_size * val_fraction)
    val_index = test_index + int(data_size * test_fraction)

    # normalizing the data (rescales the target variable to have a mean of 0 and a standard deviation of 1 - improving numerical stability)
    #computes mean and std of all training targets
    data_mean = qm9.data.y[0:train_index].mean()
    data_std = qm9.data.y[0:train_index].std()

    qm9.data.y = (qm9.data.y - data_mean) / data_std

    # datasets into DataLoader
    train_loader = DataLoader(qm9[0:train_index], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(qm9[train_index:test_index], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(qm9[test_index:val_index], batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, data_mean.numpy(), data_std.numpy()


def train_model(model, data_loader, criterion, optimizer):
    """
    Train the model for one epoch
    Args:
        model: model to be trained
        data_loader: DataLoader object containing training data
        criterion: loss function
        optimizer: optimization algorithm
    """
    model.train()
    total_loss = 0

    for d in data_loader:
        optimizer.zero_grad()
        d.x = d.x.float()
        #forward pass
        output = model(d)
        #calculate loss
        loss = criterion(output, torch.reshape(d.y, (len(d.y), 1)))

        total_loss += loss / len(data_loader)
        loss.backward()
        optimizer.step()

    return total_loss, model

def validate_model(model, data_loader, criterion):
    model.eval()
    val_loss = 0
    for d in data_loader:
        #forward pass
        output = model(d)
        #calculate loss
        loss = criterion(output, torch.reshape(d.y, (len(d.y), 1)))
        val_loss += loss / len(data_loader)

    return val_loss

@torch.no_grad()
def test_model(model, data_loader, criterion):
    """Testing"""

    test_loss = 0
    test_target = np.empty((0))
    test_y_target = np.empty((0))
    for d in data_loader:
        output = model(d)
        loss = criterion(output, torch.reshape(d.y, (len(d.y), 1)))
        test_loss += loss / len(data_loader)
        # save prediction vs ground truth values for plotting
        #Need to convert tensors to python arrays and concatenate acrosss all batches.
        test_target = np.concatenate((test_target, output.detach().numpy()[:, 0]))
        test_y_target = np.concatenate((test_y_target, d.y.detach().numpy()))

    return test_loss, test_target, test_y_target

