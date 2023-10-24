import torch
import torch.nn as nn

class Qnet(nn.Module):
    """
    Dueling DQN을 위한 Q network
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.layer1 = nn.Linear(s_dim, 128)
        self.layer2 = nn.Linear(128, 128)

        self.adv_layer = nn.Linear(128, a_dim)
        self.val_layer = nn.Linear(128, 1)
        self.act = nn.ReLU()

    def forward(self, s):
       x = self.act(self.layer1(s))
       x = self.act(self.layer2(x))

       adv = self.adv_layer(x)
       val = self.val_layer(x)
       
       adv_mean = torch.mean(adv, dim=1, keepdim=True)
       q = val + adv - adv_mean
       return q