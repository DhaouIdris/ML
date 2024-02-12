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

    def forward(self, x):
        x = torch.tanh(self.input_weights(x))
        for _ in range(100):
            x = torch.tanh(self.reservoir(x))
        return self.output_weights(x)


env = gym.make("CartPole-v1")

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

rc = RC(input_size=state_space, reservoir_size=128, output_size=action_space)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rc.parameters(), lr=0.001)


# Training
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    while True:
        
        state = torch.tensor(state[0])
        
        action_probs = rc(state)
        action = torch.argmax(action_probs).item()
        
        next_state, action, reward, done, _ = env.step(action)
        total_reward += reward
