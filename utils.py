import torch
import numpy as np 

def tensorize(array):
    tensor = torch.tensor(np.array(array).reshape(1,-1))
    tensor = tensor.float()
    return tensor

def make_batch(transition):
    x = list(zip(*transition))
    x = list(map(torch.cat, x))
    return x