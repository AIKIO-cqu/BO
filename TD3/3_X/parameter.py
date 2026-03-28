import numpy as np
import matplotlib.pyplot as plt
from Agent import TD3
import os
import argparse
import json
import torch


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    path = path + '/TD3'
    agent = TD3(path, s_dim=6)
    agent.var = 0.

    para_record = []
    for i in range(200):
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
    main()