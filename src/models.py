import torch
from torch import nn


class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)
