import torch
from torch import nn

from examples.StackedAutoencoder.pytorch.cnn_autoencoder import CNNAutoencoder
from examples.StackedAutoencoder.pytorch.denseautoencoder import DenseAutoencoder


class StackedAutoencoder(nn.Module):
    def __init__(self, hidden_dim, version=1, stack_times = 2, binary=False):
        super().__init__()
        self._version = version
        self._stack_times = stack_times
        self._binary = binary
        if version == 0:
            self.N = CNNAutoencoder(hidden_dim)
        else:
            self.N = DenseAutoencoder(hidden_dim)

    def singleAE(self, x):
        return self.N(x)

    def forward(self, x):
        for n in range(self._stack_times):
            x = self.N(x)   # first block
        if self._binary:
            # factor 10, increase discretization results to 0 and 1
            x = torch.sigmoid(10.0 * (x - 0.5))
        return x