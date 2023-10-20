import torch
import torch.nn as nn

class Qnet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.layer1 = nn.Linear(s_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, a_dim)
        self.act = nn.ReLU()

    def forward(self, s):
       x = self.act(self.layer1(s))
       x = self.act(self.layer2(x))
       a = self.layer3(x)
       return a