import torch
import torch.nn as nn

from utils import tensorize
from utils import make_batch
from network import Qnet
from buffer import Buffer
from simulator import Simulator

class Agent:
    """
    Dueling DQN Agent
    """
    def __init__(self, s_dim, a_dim):
        
        self.buffer = Buffer(5000)
        self.qnet = Qnet(s_dim, a_dim)
        self.qnet_target = Qnet(s_dim, a_dim)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.simul = Simulator(self.qnet, self.buffer)

        self.mse = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=1e-4)

    def update(self, s, a, r, ns, done):
        """
        Q learning style update
        """
        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            target = r + 0.99 * q_max * (1-done)

        q = self.qnet(s).gather(1, a.type(torch.int64))
        loss = self.mse(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Soft target update
        tau = 0.005

        for param, target_param in zip(self.qnet.parameters(), 
                                       self.qnet_target.parameters()):
            target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)

        return loss.item()

    def train(self, config):
        """
        Train loop
        """
        episode = config['episode']
        batch_size = config['batch_size']

        sell_money_ma = 0
        sell_moneys = []
        cum_rewards = []
        losses = []

        for epi in range(episode):
            sell_money, cum_reward, eps = self.simul.play_horizon(config)
            sell_money_ma += 0.01*(sell_money - sell_money_ma)

            samples = self.buffer.sample(min(batch_size, len(self.buffer)))
            batch = make_batch(samples)
            loss = self.update(*batch)

            sell_moneys.append(sell_money)
            cum_rewards.append(cum_reward)
            losses.append(loss)

            print(f'epi:{epi}')
            print(f'loss:{loss}')
            print(f'score:{sell_money_ma}')
            print(f'eps:{eps}')
            print(f'cum_reward:{cum_reward}\n')

        return sell_moneys, cum_rewards, losses
        

if __name__ == '__main__':

    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rand', type=int, default=1)
    parser.add_argument('--waiting', type=int, default=20)
    parser.add_argument('--time_cut', type=int, default=10)
    parser.add_argument('--episode', type=int, default=50000)
    parser.add_argument('--target_volume', type=int, default=30000)
    parser.add_argument('--minima_volume', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    s_dim = 25
    a_dim = 10
    rand = args.rand

    config = {
        'waiting': args.waiting,
        'time_cut': args.time_cut,
        'target_volume': args.target_volume,
        'minima_volume': args.minima_volume,
        'episode': args.episode,
        'batch_size': args.batch_size,
        }

    agent = Agent(s_dim, a_dim)
    sell_moneys, cum_rewards, losses = agent.train(config)
    
    torch.save(agent.qnet.state_dict(), 
               f'result/seed{rand}/qnet.pth')

    pd.DataFrame({'sell_money':sell_moneys}).\
        to_csv(f'result/seed{rand}/sell_money.csv')
    pd.DataFrame({'cum_rewards':cum_rewards}).\
        to_csv(f'result/seed{rand}/cum_rewards.csv')
    pd.DataFrame({'losses':losses}).\
        to_csv(f'result/seed{rand}/losses.csv')

    


