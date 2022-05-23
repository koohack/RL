from torch.distributions import Categorical, Normal
from collections import deque
from torch.autograd import Variable
import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import time
import random
import math
import numpy as np

ENV = gym.make("InvertedPendulumSwingupBulletEnv-v0")
OBS_DIM = ENV.observation_space.shape[0]
ACT_DIM = ENV.action_space.shape[0]
ACT_LIMIT = ENV.action_space.high[0]
ENV.close()
GAMMA = 0.95

## ==========================
## Parameters
## ==========================

##############################################################
############ 1. Actor Network, Critic Network 구성 ############
##############################################################

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(OBS_DIM, 512)
        self.layer2 = nn.Linear(512, 256)

        self.mu_layer = nn.Linear(256, ACT_DIM)
        self.log_std_layer = nn.Linear(256, ACT_DIM)
        self.tanh = nn.Tanh()
        self.softplu = nn.Softplus()

        self.crilayer = nn.Linear(256, 1)

    def act(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        mu = 2 * self.tanh(self.mu_layer(x))
        mu = torch.clamp(mu, -1.0, 1.0)
        std = self.softplu(self.log_std_layer(x)) + 1e-5

        return mu, std

    def cri(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        out = self.crilayer(x)
        return out

###########################################################################################
############  2. Local actor 학습(global actor, n_steps 받아와서 학습에 사용합니다.)  ############
###########################################################################################

def update(model, global_model):
    for param, global_param in zip(model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            return
        else:
            global_param._grad = param.grad

def getProb(action, mu, std):
    pi = np.array([math.pi])
    pi = torch.FloatTensor(pi)
    pi = Variable(pi)
    a = (-1 * (action - mu).pow(2) / (2 * std)).exp()
    b = 1 / (2 * std * pi.expand_as(std)).sqrt()
    return a*b

def Worker(global_actor, n_steps):
    ## Actor Critic Network
    local_actor = ActorCritic()
    local_actor.load_state_dict(global_actor.state_dict())

    ## Optimizer
    if n_steps < 3:
        optimizer = optim.Adam(local_actor.parameters(), lr=3e-5, betas=(0.95, 0.999))
    else:
        optimizer = optim.Adam(local_actor.parameters(), lr=3e-5, betas=(0.95, 0.999)) #6e-4 실험 필요

    ## Environment
    env = gym.make('InvertedPendulumSwingupBulletEnv-v0')

    ## Parameter
    pi = np.array([math.pi])
    pi = torch.FloatTensor(pi)

    ## =========================================
    ## Episodes start
    ## =========================================
    for train_episode in range(3000):
        ## update the weigth from global network
        local_actor.load_state_dict(global_actor.state_dict())

        ## start the episode
        state = env.reset()

        ## Episode total score
        score = 0

        ## Set memory
        memory = deque()

        for step in range(10001):
            ## =========================
            ## Here is one step
            ## =========================
            state = torch.FloatTensor(state)

            ## Get learnable features
            mu, std = local_actor.act(state)
            normal = Normal(mu.view(1, ).data, std.view(1, ).data)
            action = normal.sample().numpy()

            torchAction = torch.FloatTensor(action)
            act = Variable(torchAction)
            prob = getProb(act, mu, std)

            entropy = 0.5 * ((std * 2 * pi.expand_as(std)).log()+1)

            log_prob = (prob + 1e-6).log()

            ## move one action
            action = action.clip(-1, 1)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)

            ## Add memory for n-step
            memory.append([state, next_state, action, reward, log_prob, entropy])

            if done : break

            score += reward

            if n_steps == 0:
                a=0

            else:
                actor_loss = 0
                critic_loss = 0
                temp = torch.zeros(1, 1)
                lastValue = local_actor.cri(next_state)
                R = Variable(lastValue)

                ## Here start learning
                if len(memory) == n_steps:
                    for i in reversed(range(n_steps)):
                        R = GAMMA * R + memory[i][3]
                        advantage = R - local_actor.cri(memory[i][0])
                        critic_loss += 0.5 * advantage.pow(2)

                        deltaT = memory[i][3] + GAMMA * \
                                 local_actor.cri(memory[i][1]).data - local_actor.cri(memory[i][0]).data

                        temp = temp * GAMMA + deltaT

                        actor_loss = actor_loss - \
                                     (memory[i][4].sum() * Variable(temp)) - \
                                     (0.01 * memory[i][5].sum())

                    ## Train
                    optimizer.zero_grad()
                    (actor_loss + 0.5 * critic_loss).backward()
                    update(local_actor, global_actor)
                    optimizer.step()

                    ## Out one memory
                    memory.clear()
                    #memory.popleft()

            if n_steps > 0 and len(memory) > 0:
                actor_loss = 0
                critic_loss = 0
                temp = torch.zeros(1, 1)
                lastValue = local_actor.cri(memory[-1][1])
                R = Variable(lastValue)

                for i in reversed(range(len(memory))):
                    R = GAMMA * R + memory[i][3]
                    advantage = R - local_actor.cri(memory[i][0])
                    critic_loss += 0.5 * advantage.pow(2)

                    deltaT = memory[i][3] + GAMMA * \
                             local_actor.cri(memory[i][1]).data - local_actor.cri(memory[i][0]).data

                    temp = temp * GAMMA + deltaT

                    actor_loss = actor_loss - \
                                 (memory[i][4].sum() * Variable(temp)) - \
                                 (0.01 * memory[i][5].sum())

                ## Train
                optimizer.zero_grad()
                (actor_loss + 0.5 * critic_loss).backward()
                update(local_actor, global_actor)
                optimizer.step()

                ## Out one memory
                memory.clear()


            ## ===============
            ## Here one episode END
            ## ===============
            state = next_state

        print("Episode {0}, Total Score : {1}".format(train_episode, score))
        global_actor.load_state_dict(local_actor.state_dict())

    env.close()
    print("Training process reached maximum episode.")
