import torch
from torch import nn


class DistLoss(nn.Module):
    """
    NLL Loss applied to probability distribution.
    """
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight)

    def forward(self, x, l):
        return self.loss(torch.log(x), l)
