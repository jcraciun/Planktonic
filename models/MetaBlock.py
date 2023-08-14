import torch
from torch import nn


class MetaBlock(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U):
        #print("vsize", V.size())
        #print("usize", U.size())
        super(MetaBlock, self).__init__()
        self.T1 = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.T2 = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V, U):
        t1 = self.T1(U)
        t2 = self.T2(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        #print("vsize", V.size())
        return V