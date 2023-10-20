import torch
import numpy as np 
import pandas as pd 

from utils import make_batch
from utils import tensorize
from network import Qnet

class Simulator:
    def __init__(self, qnet, buffer):
        self.data = pd.read_csv('data.csv', index_col=0)
        self.data['system_time'] = pd.to_datetime(self.data['system_time'])
        self.qnet = qnet
        self.buffer = buffer

    def get_reward(self, executed_at_mid, executed_price):
        gap = executed_price - executed_at_mid
        reward = gap / 1e5
        return reward

    def get_state(self, observe):
        bid_ask_spread = observe['spread']
        mid_price = observe['midpoint']
        ask_volumes = [observe[f'ask{i}_v'] for i in range(1, 11)]
        bid_volumes = [observe[f'bid{i}_v'] for i in range(1, 11)]

        volumes = ask_volumes + bid_volumes
        ask_volume = sum(ask_volumes)
        bid_volume = sum(bid_volumes)
        imbalance = bid_volume - ask_volume
        state = [bid_ask_spread, mid_price, imbalance] + volumes
        return state

    def get_action(self, state, eps):
        state = torch.tensor(state).float()
        state = torch.unsqueeze(state, 0)

        prob = np.random.uniform(low=0.0, high=1.0, size=1)
        prob = torch.tensor(prob).float()

        if prob <= eps:
            action = np.random.choice(range(10))
            return action

        q_value = self.qnet(state).detach()
        action = q_value.argmax(dim=-1)
        action = action.squeeze(0).item()
        return action

    def action2level(self, action):
        bid_levels = list(range(1, 6))
        ask_levels = list(range(-5, 0))
        levels = bid_levels + ask_levels
        level = levels[action]
        return level

    def level2action(self, level):
        bid_levels = list(range(1, 6))
        ask_levels = list(range(-5, 0))
        levels = bid_levels + ask_levels
        action = levels.index(level)
        return action

    def get_timing(self, observe, cut):
        hour = observe['system_time'].hour
        minute = observe['system_time'].minute
        minutes = 60 * hour + minute

        if minutes % cut == 0:
            return  True
        else:
            return False

    def execution(self, price, volume, observe):
        order_volume = volume
        
        sell_tax = 0.003
        sell_fee = 0.015
        cost = sell_tax + sell_fee

        volumes = np.array([observe[f'ask{i}_v'] for i in range(1, 11)])
        prices = np.array([observe[f'ask{i}'] for i in range(1, 11)])
        indice = np.where(prices <= price)[0]    
        level = indice[-1] + 1 if len(indice) > 0 else 0

        executed_volumes = []

        executed_price = 0
        executed_volume = 0
        executed_volume = volume

        volumes_ = volumes[:level]
        prices_ = prices[:level]
                    
        for i in range(level):
            executed_v = min(order_volume, volumes_[i])
            executed_volumes.append(executed_v)
            order_volume -= executed_v

        executed_volume = sum(executed_volumes)
        remain_volume = order_volume
        executed_price = np.inner(prices_, executed_volumes) * (1-cost) 

        return executed_volume, remain_volume, executed_price
    
    def play_horizon(self, config):
        
        waiting = config['waiting']
        time_cut = config['time_cut']
        target_volume = config['target_volume']
        minima_volume = config['minima_volume']
        eps = 1.0
        cum_reward = 0
        sell_money = 0
        H = 300

        pending_orders = []

        for time in range(H):
            
            eps *= 0.99999
            done = time // (H-1)
            base_volume = min(minima_volume, target_volume) 

            observe = self.data.iloc[time].to_dict()
            observe_ = self.data.iloc[min(time+time_cut, H-1)].to_dict()

            timing = self.get_timing(observe, time_cut)

            # 대기 주문 관리 
            for _ in range(len(pending_orders)):

                order = pending_orders.pop(0)
                term = time - order['time']
                    
                price = observe['ask10'] if done else order['price']
                result = self.execution(price, order['volume'], observe)

                executed_volume = result[0] 
                remain_volume = result[1] 
                executed_price = result[2]  

                target_volume -= executed_volume
                sell_money += executed_price

                executed_at_mid = order['target'] * order['mid']
                executed_price_cum = order['executed_price_cum'] + executed_price
                
                state = self.get_state(order['observe'])
                next_state = self.get_state(order['observe_'])
                action = self.level2action(order['level'])
                reward = None

                if remain_volume == 0:
                    reward = self.get_reward(executed_at_mid, executed_price_cum)

                if term >= waiting:
                    reward = -5

                if (remain_volume > 0) & (term < waiting):

                    pending_order = {
                        'price':limit_price,
                        'volume':remain_volume,
                        'observe':order['observe'],
                        'observe_':order['observe_'],
                        'level': order['level'],
                        'target':order['target'],
                        'time':order['time'],
                        'mid':order['mid'],
                        'executed_price_cum':executed_price_cum,
                        }
                    
                    pending_orders.append(pending_order)

                if reward is not None:
                    cum_reward += reward
                    sample = [state, action, reward, next_state, done]
                    sample = list(map(tensorize, sample))
                    self.buffer.push(sample) 

            # RL에 의한 주문 생성
            if timing or done:

                state = self.get_state(observe) 
                next_state = self.get_state(observe_)

                action = self.get_action(state, eps) 
                level = self.action2level(action) if not done else 10
                
                limit_price = observe[f'ask{level}'] \
                    if level > 0 else observe[f'bid{abs(level)}']
                limit_volume = observe[f'ask{level}_v'] \
                    if level > 0 else observe[f'bid{abs(level)}_v']
                
                result = self.execution(limit_price, base_volume, observe)

                executed_volume = result[0]
                remain_volume = result[1]
                executed_price = result[2] 
                
                target_volume -= executed_volume
                sell_money += executed_price
                
                # 전량 체결 되면 transition으로 사용
                if remain_volume == 0:
                    executed_at_mid = executed_volume * observe['midpoint']
                    reward = self.get_reward(executed_at_mid, executed_price)
                    sample = [state, action, reward, next_state, done]
                    sample = list(map(tensorize, sample))
                    self.buffer.push(sample) 
                    cum_reward += reward

                # 일부 체결 되면 잔량에 대한 대기 주문 생성 
                if remain_volume > 0:
                    
                    pending_order = {
                        'time':time,
                        'observe':observe,
                        'observe_':observe_,
                        'level': level,
                        'price':limit_price,
                        'volume':remain_volume,
                        'target':base_volume,
                        'executed_price_cum':executed_price,
                        'mid':observe['midpoint']}
                    
                    pending_orders.append(pending_order)

        return sell_money, cum_reward