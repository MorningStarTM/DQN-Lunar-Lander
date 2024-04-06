import torch.nn as nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random


GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_feature = int(np.prod(env.observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_feature, 65), 
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        pass


