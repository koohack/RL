import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, defaultdict
from torch.autograd import Variable
from functools import reduce


def get_demo_traj():
    return np.load("./demo_traj_2.npy", allow_pickle=True)


##########################################################################
############                                                  ############
############                  DQfDNetwork 구현                 ############
############                                                  ############
##########################################################################
class DQfDNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(DQfDNetwork, self).__init__()
        ## TODO
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(in_size, 50)
        #self.layer2 = nn.Linear(50, 30)
        self.outLayer = nn.Linear(50, out_size)

    def forward(self, x):
        ## TODO
        x = self.relu(self.layer1(x))
        #x = self.relu(self.layer2(x))
        return self.outLayer(x)

##########################################################################
############                                                  ############
############                  DQfDagent 구현                   ############
############                                                  ############
##########################################################################

class DQfDAgent(object):
    def __init__(self, env, use_per, n_episode):
        self.n_EPISODES = n_episode
        self.env = env
        self.use_per = use_per

        self.demo = get_demo_traj()
        self.demoStore = deque()
        self.democheck = defaultdict(list)
        self.mem = deque(maxlen=256)

        self.gamma = 0.95
        self.eGreedy = 0.015
        self.margin = 0.8
        self.nstep = 10

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.predmodel = DQfDNetwork(4, 2).to(self.device)
        self.fixedQ = DQfDNetwork(4, 2).to(self.device)

        self.optimizer = optim.Adam(self.predmodel.parameters(), lr=0.00002, weight_decay=1e-5) ## For preventing overfitting (L2 regulazation)

    def get_action(self, state):

        state = torch.Tensor([state]).to(self.device)
        output = self.predmodel(state).to(self.device)

        action = int(torch.argmax(output))
        if np.random.rand(1) < self.eGreedy:
            action = np.random.randint(0, 2)
            if action == 0: action = 1
            else: action = 0

        return action

    def sampleDemo(self, numSize):
        batch = random.sample(self.demoStore, numSize)

        states, actions, rewards, next_states, dones, info = [], [], [], [], [], []
        for state, action, reward, next_state, done, _info in batch:
            states.append(state)
            actions.append([action])
            rewards.append(reward)
            next_states.append(next_state)
            dones.append([done])
            info.append(_info)

        return states, actions, rewards, next_states, dones, info

    def update_preTrain(self, size):
        for loop in range(size):
            ## Sampling
            sample = random.sample(self.demoStore, 1)

            state, action, reward, next_state, done, info = sample[0]

            state = torch.Tensor(np.array([state])).to(self.device)
            action = torch.Tensor(np.array([action])).to(self.device)
            action = action.type(torch.int64)
            next_state = torch.Tensor(np.array([next_state])).to(self.device)

            ## ----------
            ## one step
            ## ----------
            predQ = self.predmodel(state)
            predictedQ = torch.gather(predQ, 1, action.unsqueeze(-1))

            expectQ = reward + self.fixedQ(next_state).max(1)[0]

            lossOne = F.mse_loss(predictedQ, torch.Tensor([expectQ]))

            ## ----------
            ## n step
            ## ----------
            epi, step = info

            s, a, r, ns, d, i = zip(*self.democheck[epi][step:step + 10])
            length = len(s)

            nStepQ = 0.0
            mul = 1.0
            for i in range(length):
                nStepQ += mul * r[i]
                mul = mul * self.gamma
            nStepQ += mul * self.fixedQ(torch.Tensor([ns[-1]])).max(1)[0]

            lossN = F.mse_loss(predictedQ, torch.Tensor([nStepQ]))

            ## ------------
            ## expert loss
            ## ------------
            aE = int(action)
            maxValue = torch.tensor(-9999999999)

            for a in range(2):
                a = torch.Tensor([[a]])
                a = a.type(torch.int64)
                value = torch.gather(predQ, 1, a) + self.Lfunction(aE, a)
                if maxValue < value[0]: maxValue = value[0]

            aE = torch.Tensor([[aE]])
            aE = aE.type(torch.int64)
            expertQ = torch.gather(predQ, 1, aE)

            lossE = F.mse_loss(maxValue, expertQ[0])

            ## -------------
            ## apply loss function
            ## -------------
            self.optimizer.zero_grad()
            totalLoss = lossOne + lossN + lossE
            totalLoss.backward()
            self.optimizer.step()

            if loop % 5 == 0: self.fixedQ.load_state_dict(self.predmodel.state_dict())

    def pretrain(self):
        ## Do pretrain for 1000 steps
        ## Check time
        start = time.time()

        ## Just store the demo data
        for epiNum, epi in enumerate(self.demo):
            for stepNum, step in enumerate(epi):
                state, action, reward, next_state, done = step
                self.democheck[epiNum].append([state, action, reward, next_state, done, (stepNum)])
                self.demoStore.append([state, action, reward, next_state, done, (epiNum, stepNum)])

        ## Pre-train 1000 times
        for loop in range(1001):
            self.update_preTrain(250)

            if loop % 100 == 0: print("{0} loop end".format(loop))

        print("-----Time : {0}".format(time.time()-start))

    def Lfunction(self, aE, a):
        return 0.0 if aE == a else 0.8

    def train(self):
        ###### 1. DO NOT MODIFY FOR TESTING ######
        test_mean_episode_reward = deque(maxlen=20)
        test_over_reward = False
        test_min_episode = np.inf
        ###### 1. DO NOT MODIFY FOR TESTING ######

        # Do pretrain
        self.fixedQ.load_state_dict(self.predmodel.state_dict())
        print("---------Pre-train Start--------")
        self.pretrain()
        print("---------Pre_train End----------")
        ## TODO

        ## draw graph
        number = []
        meanScore = []
        z = 0

        for e in range(self.n_EPISODES):
            ########### 2. DO NOT MODIFY FOR TESTING ###########
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########

            ## TODO
            done = False
            state = self.env.reset()
            #self.env.render()
            mem = deque() ## For n step
            totalmem = deque(maxlen=200)

            score = 0
            count = 0
            while not done:
                ## TODO
                action = self.get_action(state)

                ## TODO
                next_state, reward, done, _ = self.env.step(action)
                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward
                ########### 3. DO NOT MODIFY FOR TESTING  ###########

                ## Store memory
                mem.append([state, action, reward, next_state, done])
                totalmem.append([state, action, reward, next_state, done])

                ## Update as n step
                if len(mem) == 10:
                    count += 1
                    ## ------------
                    ## one step
                    ## ------------
                    state = mem[0][0]
                    action = mem[0][1]
                    reward = mem[0][2]
                    ns = mem[0][3]
                    done = mem[0][4]

                    state = torch.Tensor(np.array([state])).to(self.device)
                    action = torch.Tensor(np.array([action])).to(self.device)
                    action = action.type(torch.int64)
                    ns = torch.Tensor(np.array([ns])).to(self.device)

                    predQ = self.predmodel(state)
                    predictedQ = torch.gather(predQ, 1, action.unsqueeze(-1))

                    expectQ = reward + self.fixedQ(ns).max(1)[0]

                    lossOne = F.mse_loss(predictedQ, torch.Tensor([expectQ]))


                    ## -------------
                    ## n step
                    ## -------------
                    nStepQ = 0.0
                    mul = 1.0
                    for i in range(10):
                        nStepQ += mul * mem[i][2]
                        mul = mul * self.gamma
                    nStepQ += mul * self.fixedQ(torch.Tensor([mem[-1][3]])).max(1)[0]

                    lossN = F.mse_loss(predictedQ, torch.Tensor([nStepQ]))

                    ## ------------
                    ## expert loss
                    ## ------------
                    aE = int(action)
                    maxValue = torch.tensor(-9999999999)

                    for a in range(2):
                        a = torch.Tensor([[a]])
                        a = a.type(torch.int64)
                        value = torch.gather(predQ, 1, a) + self.Lfunction(aE, a)
                        if maxValue < value[0]: maxValue = value[0]

                    aE = torch.Tensor([[aE]])
                    aE = aE.type(torch.int64)
                    expertQ = torch.gather(predQ, 1, aE)

                    lossE = F.mse_loss(maxValue, expertQ[0])

                    self.optimizer.zero_grad()
                    totalLoss = lossOne + lossN + lossE
                    totalLoss.backward()
                    self.optimizer.step()

                    self.update_preTrain(20)

                    count +=1
                    if count % 5 == 0: self.fixedQ.load_state_dict(self.predmodel.state_dict())
                    mem.popleft()

                score += reward

                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward) == 20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    number.append(z)
                    meanScore.append(np.mean(test_mean_episode_reward))
                    z += 1

                state = next_state

                ## TODO

            print("------Now episode : {0}".format(e))
            print("------Now score : {0}".format(score-9))
            print("===========================")
            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if test_over_reward:
                print("END train function")
                break
            ########### 5. DO NOT MODIFY FOR TESTING  ###########


        #self.env.close()
        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########
