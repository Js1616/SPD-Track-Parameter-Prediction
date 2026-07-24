import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
    global_max_pool
)




import torch.nn.functional as F
class trackGAT(nn.Module):

    def __init__(self,in_features=7, hidden_dim=128, out_features=3, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_features, hidden_dim,heads=1,concat=False )
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim* 2,64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64,out_features)
        )


    def forward(self, x,edge_index, batch):
        x = self.gat1(x,edge_index)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)

        x = torch.relu(x)

        x1 = global_mean_pool(x,batch )
        x2 = global_max_pool( x,batch)

        x = torch.cat([x1,x2],dim=1)

        x = self.linear(x)
        return x