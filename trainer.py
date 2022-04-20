import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import math

from . import *
from .memory import ReplayMemory
from .model import SimpleMLP


class DQNTrainer:
    def __init__(
        self,
        config
    ):
        self.config = config
        self.env = gym.make(config.env_id)
        self.epsilon = self.config.eps_start  # My little gift for you
        self.trained_steps = self.config.train_freq - 1
        self.mem = ReplayMemory()

        self.predmodel = SimpleMLP(4, 2)
        self.fixedModel = SimpleMLP(4, 2)
        self.fixedModel.load_state_dict(self.predmodel.state_dict())
        self.optimizer = self.config.optim_cls(self.predmodel.parameters(), self.config.optim_kwargs["lr"])




