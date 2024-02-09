import gym
import torch
import torch.nn as nn
import numpy as np

class RC(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size):
        super(RC, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        
        self.input_weights = nn.Linear(input_size, reservoir_size)
        self.reservoir = nn.Linear(reservoir_size, reservoir_size, bias=False)
        self.output_weights = nn.Linear(reservoir_size, output_size)

env = gym.make("CartPole-v1")
