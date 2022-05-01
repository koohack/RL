import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import random

from . import *
from .memory import ReplayMemory
from .model import SimpleMLP


class DQNTrainer:
    def __init__(
        self,
        config
    ):
        ## Train Factor
        self.config = config
        self.env = gym.make(config.env_id)
        self.epsilon = self.config.eps_start  # My little gift for you

        ## Model Information
        self.predmodel=SimpleMLP(4, 2)
        self.fixmodel = SimpleMLP(4, 2)
        self.fixmodel.load_state_dict(self.predmodel.state_dict())
        self.savemodel = SimpleMLP(4, 2)

        ## Optimizer option learning rate = 0.0003
        self.optimizer = self.config.optim_cls(self.predmodel.parameters(), 0.0003)

        ## Replay Memory
        self.mem=ReplayMemory()

        ## Mode Information
        print("*** Select the Mode ***")
        print("*** Only Number Please ***")
        print("1. Original DQN")
        print("2. N-Step DQN")
        self.mode=int(input("Mode : "))
        self.n = 0
        if self.mode == 2: self.n = int(input("Step Number : "))
    
    def train(self, num_train_steps: int):

        episode_rewards = []
        ## Train According to the Mode
        if self.mode == 1: episode_rewards = self.Origin_DQN(num_train_steps)
        elif self.mode == 2: episode_rewards = self.NStep_DQN(num_train_steps)

        ''''' No Need
                    #if self.config.verbose:
                     #   status_string = f"{self.config.run_name:10}, Whatever you want to print out to the console"
                      #  print(status_string + "\r", end="", flush=True)
        '''''
        return episode_rewards



    # Update online network with samples in the replay memory. 
    def update_network(self):
        pass
    
    # Update the target network's weights with the online network's one. 
    def update_target(self):
        pass

    
    # Update epsilon over training process.
    def update_epsilon_Origin(self, eGreedy):
        eGreedy -= 0.0003
        eGreedy = max(self.config.eps_end, eGreedy)
        return eGreedy

    def updata_epsilon_NSTEP(self, eGreedy):
        eGreedy -= 0.00001
        eGreedy = max(self.config.eps_end, eGreedy)
        return eGreedy

    def Origin_DQN(self, num_train_steps):
        ## Maximum step setting
        maxStep = num_train_steps
        maxStep = 100000

        ## episode and score per episode
        episode = 1
        scoreList = []

        ## eGreedy option
        eGreedy = self.config.eps_start
        eGreedyMin = self.config.eps_end

        ## save the optimal model, tp -> store the 30 data, mx -> maximum average of tp
        tp = []
        mx = 0

        ## Train until maxStep smaller than zero
        while maxStep > 0:
            ## New Episode
            state = self.env.reset(seed=random.randint(1,10000))

            score = 0
            for step in range(501):
                getAction=torch.Tensor(np.array([state]))

                state=torch.Tensor(state)

                ## eGreedy based select action
                action = -1
                if random.random() < eGreedy: action = random.randint(0, 1)
                else: action = int(self.predmodel(getAction).max(1)[1])

                ## Next State
                next_state, reward, done, info = self.env.step(action)

                ## Episode is over, give reward -2
                if done : reward = -2

                ## Add replay memory
                self.mem.write(state.numpy(), action, reward, next_state, done)

                ## Learn DQN
                if len(self.mem) > self.config.batch_size+10:
                    states, actions, rewards, next_states, dones = self.mem.sample(self.config.batch_size)

                    states = torch.Tensor(np.array(states))
                    actions = torch.Tensor(np.array(actions))
                    actions = actions.type(torch.int64)
                    rewards = torch.Tensor(rewards)
                    next_states = torch.Tensor(np.array(next_states))
                    dones = torch.Tensor(dones)

                    predQ = self.predmodel(states)
                    predictedQ = torch.gather(predQ, 1, actions.unsqueeze(-1))

                    fixQ = self.fixmodel(next_states).max(1)[0]
                    expectQ = rewards + (self.config.discount_rate * fixQ)

                    loss = F.mse_loss(predictedQ.squeeze(), expectQ)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                ## Add score
                score+=1

                ## Decrease the maxStep
                maxStep-=1

                ## Update the fixed model every 30 steps
                if maxStep % 30 == 0: self.fixmodel.load_state_dict(self.predmodel.state_dict())

                ## Change the state
                state = next_state

                ## Finish the episode
                if done:
                    print("에피소드 : {0}, 점수 : {1}, 남은 Step : {2}".format(episode+1, score, maxStep))
                    scoreList.append(score)
                    tp.append(score)
                    break

            ## Increase the episode
            episode += 1
            eGreedy = self.update_epsilon_Origin(eGreedy)
            eGreedy = max(eGreedyMin, eGreedy)

            if episode % 10 == 0:
                ## Save the best model
                num = sum(tp) / len(tp)
                if num > mx and tp[-1] > 150:
                    mx = num
                    tp=[]
                    self.savemodel.load_state_dict(self.predmodel.state_dict())
                else: tp = []

        ## Final test
        self.finalTest()

        return scoreList

    def NStep_DQN(self, num_train_steps):
        ## Maximum step setting
        maxStep = num_train_steps
        maxStep = 100000

        ## episode and score per episode
        episode = 1
        scoreList = []

        ## eGreedy option
        eGreedy = self.config.eps_start
        eGreedyMin = self.config.eps_end

        ## save the optimal model, tp -> store the 30 data, mx -> maximum average of tp
        tp = []
        mx = 0

        ## Train until maxStep smaller than zero
        while maxStep > 0:
            ## New Episode
            state = self.env.reset(seed=random.randint(1,10000))

            score = 0
            for step in range(501):
                getAction = torch.Tensor(np.array([state]))

                state = torch.Tensor(state)

                ## eGreedy based select action
                action = -1
                if random.random() < eGreedy:
                    action = random.randint(0, 1)
                else:
                    action = int(self.predmodel(getAction).max(1)[1])

                ## Next State
                next_state, reward, done, info = self.env.step(action)

                ## Episode is over, give reward -2
                if done: reward = -1

                ## Add replay memory
                self.mem.write(state.numpy(), action, reward, next_state, done)

                ## Update the DQN
                if len(self.mem) == self.n+1:
                    ## FIFO first in memory
                    needUpdate = self.mem.nStep_sample()
                    temp = needUpdate.popleft()

                    ## make data
                    updateState = temp[0]
                    updateAction = temp[1]
                    updateReward = temp[2]
                    updateNextState = temp[3]
                    updateDone = temp[4]

                    updateState = torch.Tensor(np.array([updateState]))
                    updateAction = torch.Tensor(np.array([updateAction]))
                    updateAction = updateAction.type(torch.int64)
                    updateNextState = torch.Tensor(np.array([updateNextState]))

                    ## Prediction
                    predQ = self.predmodel(updateState)
                    predictedQ = torch.gather(predQ, 1, updateAction.unsqueeze(-1))

                    ## N-Step learning
                    nStepQ=0
                    gamma = self.config.discount_rate
                    mul = 1
                    for i in range(self.n):
                        nStepQ+=needUpdate[i][2] * mul
                        mul = mul * gamma
                    nStepQ += mul * self.fixmodel(torch.Tensor(np.array([needUpdate[-1][0]]))).max(1)[0]

                    ## Back propagation
                    loss = F.mse_loss(predictedQ.squeeze(), torch.Tensor([nStepQ]))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                ## Increase the score
                score += 1

                ## Decrease the maxStep
                maxStep -= 1

                ## Updata eGreedy
                eGreedy = self.updata_epsilon_NSTEP(eGreedy)

                ## Updata the fixed model
                if maxStep % 30 == 0: self.fixmodel.load_state_dict(self.predmodel.state_dict())

                ## Change state to next state
                state = next_state

                ## Episode is over
                if done:
                    print("에피소드 : {0}, 점수 : {1}, 남은 Step : {2}".format(episode + 1, score, maxStep))
                    scoreList.append(score)
                    tp.append(score)
                    self.finishLearning()
                    break

            ## Update the episode
            episode += 1

            ## Save the best model
            if episode % 12 == 0:
                num = sum(tp) / len(tp)
                if num > mx and tp[-1] > 150:
                    mx = num
                    tp = []
                    self.savemodel.load_state_dict(self.predmodel.state_dict())
                else:
                    tp = []

        ## Final test
        self.finalTest()

        return scoreList

    def finalTest(self):
        answer = []
        for i in range(5):
            seed = random.randint(1, 100)
            for episode in range(1, 11):
                done = False

                point = 0

                state = self.env.reset(seed=seed)
                while not done:
                    getAction = torch.Tensor(np.array([state]))
                    state = torch.Tensor(state)

                    action = int(self.savemodel(getAction).max(1)[1])
                    next_state, reward, done, info = self.env.step(action)

                    point += 1
                    state = next_state

                answer.append(point)
        print("Final test average score : {0}".format(sum(answer) / len(answer)))

    def finishLearning(self):
        while self.mem:
            temp = self.mem.popleft()

            st = temp[0]
            ac = temp[1]
            re = temp[2]
            nest = temp[3]

            st = torch.Tensor(np.array([st]))
            ac = torch.Tensor(np.array([ac]))
            ac = ac.type(torch.int64)
            nest = torch.Tensor(np.array([nest]))

            preQ = self.predmodel(st)
            preddQ = torch.gather(preQ, 1, ac.unsqueeze(-1))

            nStepQ = 0
            gamma = 0.98
            mul = 1
            for i in range(len(self.mem)):
                nStepQ += self.mem[i][2] * mul
                mul = mul * gamma

            if len(self.mem) > 0:
                nStepQ += mul * self.fixmodel(torch.Tensor(np.array([self.mem[-1][0]]))).max(1)[0]

            losse = F.mse_loss(self.preddQ.squeeze(), torch.Tensor([nStepQ]))
            self.optimizer.zero_grad()
            losse.backward()
            self.optimizer.step()