import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import math
import numpy as np
from copy import deepcopy as dc
import random as rd
import gym
import matplotlib.pyplot as plt

DEVICE = 'cpu'

learning_rate = 0.0001
gamma = 0.95
epsilon, epm, epd = 1, 0.01, 0.001
X, cX = 10, 0

env = gym.make("CartPole-v1")

N_EPISODES = 1000
MEMORY_SIZE = 10000
BATCH_SIZE = 64
memory_count = 0
replaystates = np.zeros((MEMORY_SIZE, *env.observation_space.shape), dtype=np.float32)
replayactions = np.zeros(MEMORY_SIZE, dtype=np.int64)
replayrewards = np.zeros(MEMORY_SIZE, dtype=np.float32)
replaystates_ = np.zeros((MEMORY_SIZE, *env.observation_space.shape), dtype=np.float32)
replaydones = np.zeros(MEMORY_SIZE, dtype=np.bool)

def update_epsilon():
    global epsilon
    epsilon = max(epm, epsilon * math.exp(-epd))

def save_replay(state, action, reward, state_, done):
    global memory_count
    index = memory_count % MEMORY_SIZE
    replaystates[index] = state
    replayactions[index] = action
    replayrewards[index] = reward
    replaystates_[index] = state_
    replaydones[index] = 1 - done

    memory_count+=1 


def training_batch():
    memsize = min(memory_count, MEMORY_SIZE)
    batch_indices = np.random.choice(memsize, BATCH_SIZE, replace=True)
    
    states = replaystates[batch_indices]
    actions = replayactions[batch_indices]
    rewards = replayrewards[batch_indices]
    states_ = replaystates_[batch_indices]
    dones = replaydones[batch_indices]

    return states, actions, rewards, states_, dones

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 2)

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.loss_function = nn.MSELoss()
        self.to(DEVICE)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

best_score = 0
average_score = 0
avg_scores = []
x_axis = []

for i in range(1, N_EPISODES + 1):
    current_state = env.reset()[0]
    current_state = np.reshape(current_state, [1, env.observation_space.shape[0]])
    score = 0

    while True:
        action = choose_action(current_state)
        new_state, reward, done, trunc, _ = env.step(action)
        new_state = np.reshape(new_state, [1, env.observation_space.shape[0]])
        score+=reward
        save_replay(current_state, action, reward, new_state, done or trunc)
        learn()

        if done or trunc:
            if score > best_score: 
                best_score = score
            average_score += score
            avg_scores.append(average_score/i)
            x_axis.append(i)

            
            print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_score/i, best_score, score, epsilon))

            break

        current_state = new_state

