import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import RollControlEnv
from Agent import DDPG, SAC, PPO, TD3
import os
import argparse
import json
import torch


def main(method):
    path = os.path.dirname(os.path.realpath(__file__))
    path = path + '/Record_' + method
    agent = None
    if method == 'TD3':
        agent = TD3(path, s_dim=2)
        agent.var = 0.
    elif method == 'DDPG':
        agent = DDPG(path, s_dim=5)
        agent.var = 0.
    elif method == 'SAC':
        agent = SAC(path, s_dim=5)
        agent.test = True
    elif method == 'PPO':
        agent = PPO(path, s_dim=5)
        agent.test = True
    else:
        print('somthing wrong')
        exit()

    para_record = []
    for i in range(500):
        print(i)
        agent.load_net(prefix=str(i))
        parameter = torch.tensor(0.)
        for param in agent.actor.parameters():
            parameter += torch.sum(torch.abs(param.detach()))
        para_record.append(parameter.detach().item())

    index = np.array(range(len(para_record)))
    plt.plot(index, para_record, label='x')
    plt.show()


if __name__ == '__main__':
    main('TD3')