from . import *
from typing import Tuple
from collections import deque
import numpy as np
import random

class ReplayMemory:
    def __init__(
        self, 
        observation_shape: tuple = (),
        action_shape: tuple = (),
        buffer_size: int = 50000,
        num_steps: int = 1,
    ):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.num_steps = num_steps

        self.mem = deque()

    def write(self, state, action, reward, next_state, done):
        self.mem.append([state, action, reward, next_state, done])

    # 버퍼에서 Uniform Random으로 Transition들을 뽑습니다.
    def sample(self, num_samples: int = 1): # -> Tuple[np.ndarray]:
        batch = random.sample(self.mem, num_samples)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return [states, actions, rewards, next_states, dones]

    def nStep_sample(self):
        return self.mem

    def clean(self):
        self.mem=deque()

    def __len__(self):
        return len(self.mem)