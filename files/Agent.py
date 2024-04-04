import torch.nn as nn
import torch.nn.functional as F
import torch as T
import gym
from dataclasses import dataclass
from typing import Any
from random import sample

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class DQNAgent:
    def __init__(self, model):
        self.model = model

    def get_actions(self, observations):
        q_vals = self.model(observations)
        return q_vals.max(-1)
    

class Model(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(Model, self).__init__()
        assert len(obs_shape) == 1, "This network only works for flat observations"
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )
        self.opt = T.optim.Adam(self.net.parameters(), lr=0.0001)

    def forward(self, x):
        return self.net(x)
    




    

class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = []

    def insert(self, sars):
        self.buffer.append(sars)
        self.buffer = self.buffer[-self.buffer_size:]

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)
    

def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())



def train_step(model, state_transitions, tgt, num_actions):
    cur_states = T.stack([s.state for s in state_transitions])
    rewards = T.stack([s.reward for s in state_transitions])
    mask = T.stack([0 if s.done else 1 for s in state_transitions])
    next_states = T.stack([s.next_state for s in state_transitions])
    actions = T.stack([s.action for s in state_transitions])

    with T.no_grad():
        qvals_next = tgt(next_states).max(-1)

    model.optimizer.zero_grad()
    qvals = model(cur_states)
    one_hot_actions = F.one_hot(T.LongTensor(action, num_actions))

    loss = (rewards + qvals_next - T.sum(qvals*one_hot_actions, -1)).mean()
    loss.backwards()
    model.optimizer.step()
    return loss

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    observation = env.reset()
    obs_ = observation[0]

    m = Model(env.observation_space.shape, env.action_space.n)
    tgt = Model(env.observation_space.shape, env.action_space.n)

    rb = ReplayBuffer()
    #q_vals = m(T.Tensor(obs_))
    try:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)

        rb.insert(Sarsd(obs_, action, reward, observation, done))
        obs_ = observation

        if done:
            observation = env.reset()

        if len(rb.buffer) > 5000:
            import ipdb; ipdb.set_trace()
    except KeyboardInterrupt:
        pass
    env.close()

    print(rb.buffer[0])
