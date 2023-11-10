
import math
import numpy as np
import gym
import matplotlib.pyplot as plt

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
