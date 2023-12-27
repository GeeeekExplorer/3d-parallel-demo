import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(in_dim, hid_dim) * (2 / in_dim) ** 0.5)
        self.w2 = nn.Parameter(torch.randn(hid_dim, out_dim) * (2 / hid_dim) ** 0.5)
    
    def forward(self, x: torch.Tensor):
        return (x @ self.w1).relu() @ self.w2
