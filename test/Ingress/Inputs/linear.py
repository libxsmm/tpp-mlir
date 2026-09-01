import torch
import torch.nn as nn

M = 64
N = 64
K = 64

class Model(nn.Module):
    def __init__(self, M=M, N=N, K=K, bias=False, layers=1, relu=False, rand=False, trans_a=False, trans_b=False):
        super().__init__()
        self.layers = layers
        self.relu = relu
        self.trans = trans_a
        self.fc = nn.Linear(K, N, bias=bias)
        if not rand:
          with torch.no_grad():
              self.fc.weight.fill_(1.0)
              if bias:
                  self.fc.bias.fill_(0.0)
        if trans_b:
            self.fc.weight = nn.Parameter(torch.transpose(self.fc.weight, 0, 1))

    def forward(self, x):
        for _ in range(self.layers):
            if self.trans:
                x = torch.transpose(x, 0, 1)
            x = self.fc(x)
            if self.relu:
                x = torch.relu(x)
        return x

def get_init_inputs():
    return [M, N, K, False, 1, False, False, False, False]

def get_inputs():
    return (torch.ones(M, K),)
