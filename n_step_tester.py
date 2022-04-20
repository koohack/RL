import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import distutils.spawn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import model
import memory

env = gym.make('CartPole-v1')

mem=memory.ReplayMemory()

predmodel=model.SimpleMLP(4, 2)
fixmodel=model.SimpleMLP(4, 2)
fixmodel.load_state_dict(predmodel.state_dict())

optimizer=optim.Adam(predmodel.parameters(), 0.002)

scoreList=[]

eGreedy=0.3

maxStep=100000