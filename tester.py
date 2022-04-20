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

for episode in range(1000):## episode
    state = env.reset()

    score=0
    for j in range(503):
        env.render()

        getAction=torch.Tensor(np.array([state]))

        state=torch.Tensor(state)

        ## argmax
        action=-1
        if random.random() < eGreedy: action=random.randint(0, 1)
        else: action=int(predmodel(getAction).max(1)[1])

        next_state, reward, done, info=env.step(action)

        if done: reward = -1

        ## add replay memory
        mem.write(state.numpy(), action, reward, next_state, done)

        ## learn the deep q network
        if len(mem) > 520 :
            states, actions, rewards, next_states, dones = mem.sample(512)

            states=torch.Tensor(np.array(states))
            actions=torch.Tensor(np.array(actions))
            actions=actions.type(torch.int64)
            rewards=torch.Tensor(rewards)
            next_states=torch.Tensor(np.array(next_states))
            dones=torch.Tensor(dones)

            predQ = predmodel(states)
            predictedQ = []
            #for num in range(512):
             #   predictedQ.append(predQ[num][actions[num]])
            #predictedQ=torch.Tensor(predictedQ)
            #print(predQ)
            #print(actions)
            predictedQ = torch.gather(predQ, 1, actions.unsqueeze(-1))

            fixQ = fixmodel(next_states).max(1)[0]
            expectQ = rewards + (0.99*fixQ)
            #print(predictedQ)
            #print("--------")
            #print(expectQ)

            loss = F.mse_loss(predictedQ.squeeze(), expectQ)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        score += reward
        state = next_state

        if done or j==502:
            print("에피소드 : {0}, 점수 : {1}, 남은 Step : {2}".format(episode+1, score, maxStep))
            scoreList.append(score)
            break

    maxStep-=score
    if maxStep < 0:
        break

    eGreedy-=0.0003
    eGreedy=max(0.01, eGreedy)
    if episode % 50 == 0: fixmodel.load_state_dict(predmodel.state_dict())

plt.plot(scoreList)
plt.show()