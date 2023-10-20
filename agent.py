import torch
import torch.nn as nn

from utils import tensorize
from utils import make_batch
from network import Qnet
from buffer import Buffer
from simulator import Simulator

class Agent:
    def __init__(self, s_dim, a_dim):
        
        self.buffer = Buffer(5000)
        self.qnet = Qnet(s_dim, a_dim)
        self.qnet_target = Qnet(s_dim, a_dim)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.simul = Simulator(self.qnet, self.buffer)

        self.mse = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=1e-4)

    def update(self, s, a, r, ns, done):

        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            target = r + 0.99 * q_max * (1-done)

        q = self.qnet(s).gather(1, a.type(torch.int64))
        loss = self.mse(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def train(self, config):

        episode = config['episode']
        batch_size = config['batch_size']

        for _ in range(episode):
            sell_money, cum_reward = self.simul.play_horizon(config)

            if len(self.buffer) > batch_size:
                samples = self.buffer.sample(len(self.buffer))
                batch = make_batch(samples)
                loss = self.update(*batch)

                print(loss, cum_reward)



if __name__ == '__main__':
    s_dim = 23
    a_dim = 10

    config = {
        'waiting': 20,
        'time_cut': 10,
        'target_volume': 30000,
        'minima_volume': 1000,
        'episode': 10000,
        'batch_size': 64,
        }

    agent = Agent(s_dim, a_dim)
    agent.train(config)
