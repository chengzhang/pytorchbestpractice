import numpy as np
import torch
from torch import nn


class MyDropout(nn.Module):
    def __init__(self, drop_ratio=0.8):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.mask = None

    def forward(self, X, train_flag):
        if train_flag:
            self.mask = np.random.rand(*X.shape) > self.drop_ratio
            return X * self.mask
        return X * (1 - self.drop_ratio)

    def backward(self, grad):
        return grad * self.mask
