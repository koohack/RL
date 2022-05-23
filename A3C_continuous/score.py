import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.multiprocessing as mp
from torch.distributions import Normal
import matplotlib.pyplot as plt
import torch
import time
import gym
import argparse
from agent import Worker, ActorCritic
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

###########################################################################################
############  3. Evaluate하는 부분(목표 reward에 도달하면 조기종료)  #############################
###########################################################################################

def Evaluate(global_actor, mode):
    env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
    score = 0.0
    epi_rew = []
    start_time = time.time()

    for n_epi in range(300001):
        done = False
        s = env.reset()
        finish = False
        while not done:
            mu, std = global_actor.act(torch.from_numpy(s).float())
            norm_dist = Normal(mu, std)
            a = norm_dist.sample()
            s_prime, r, done, _ = env.step(a)
            s = s_prime
            score += r

        if n_epi % 100 == 0 and n_epi != 0:
            mean_reward = score / 100
            epi_rew.append(mean_reward)
            print("Episode: {}, avg score: {:.1f}".format(n_epi, mean_reward))
            if mode == "SS" and mean_reward >= 400:
                finish = True
                print("Solved (1)!!!, Time : {:.2f}".format(time.time() - start_time))
            elif mode == "MS" and mean_reward >= 500:
                finish = True
                print("Solved (2)!!!, Time : {:.2f}".format(time.time() - start_time))
            elif mode == "MM" and mean_reward >= 600:
                finish = True
                print("Solved (3)!!!, Time : {:.2f}".format(time.time() - start_time))
            score = 0.0
            time.sleep(1)

        if finish:
            break
        
###########################################################################################
#########################  4. Evaluate 끝나고 plot 그리는 부분  ###############################
###########################################################################################
    plt.title(mode)
    plt.plot(np.arange(len(epi_rew)), epi_rew, label=mode)
    plt.xlabel('Episode * 100')
    plt.ylabel('Rewards')
    
    plt.savefig('plot_%s.png' % mode)
    plt.close()
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate Your Actor Critic Model")
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--multi', type=int, default=1)
    args = parser.parse_args()

    if args.n_steps == 1 and args.multi == 1:
        mode = "SS"
    elif args.n_steps != 1 and args.multi == 1:
        mode = "MS"
    elif args.n_steps != 1 and args.multi !=1:
        mode = "MM"
    else: mode = "MS"

    global_actor = ActorCritic()
    global_actor.share_memory()

    processes = []
    for rank in range(args.multi + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=Evaluate, args=(global_actor, mode))
        else:
            p = mp.Process(target=Worker, args=(global_actor, args.n_steps,))

        p.start()
        processes.append(p)

    for p in processes:
        p.join()
