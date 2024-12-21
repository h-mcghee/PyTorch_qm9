import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as Fun

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()

        #layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, e = data.x, data.edge_index

        #activation function - replaces negative values with zeros. 
        x = self.conv1(x,e)
        x = x.relu()

        x = self.conv2(x,e)
        x = x.relu()

        x = self.conv3(x,e)
        x = global_mean_pool(x,data.batch)

        x = Fun.dropout(x, p = 0.5, training = self.training)

        out = self.lin(x)
    
        return out
