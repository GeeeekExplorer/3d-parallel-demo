import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden_dim) / in_dim)
        self.w2 = nn.Parameter(torch.randn(hidden_dim, out_dim) / hidden_dim)
    
    def forward(self, x: torch.Tensor):
        return (x @ self.w1).relu() @ self.w2
