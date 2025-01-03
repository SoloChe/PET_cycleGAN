import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image
import argparse

#### use this for bool parameter in argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
    
class CosineAnnealingLR_with_Restart_WeightDecay:
    def __init__(self, optimizer, T_max, T_mult=1, eta_min=0, eta_max=0.1, decay=0.9):
        self.optimizer = optimizer
        self.decay = decay
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_cur = 0

    def step(self):
        self.T_cur += 1
        if self.T_cur >= self.T_max:
            self.T_cur = 0
            self.eta_max *= self.decay
            self.T_max *= self.T_mult
            
            
        lr = self.eta_min + (0.5 * (self.eta_max-self.eta_min) * (1 + np.cos(np.pi * self.T_cur / self.T_max)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            

