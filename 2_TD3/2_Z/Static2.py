import time
from EnvUAV.env import YawControlEnv
from Agent import TD3
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/TD3_'
    with open(path + '0.98/disr0.98.json', 'r') as f:
        reward_store3 = json.load(f)

    reward_store3 = np.array(reward_store3)

    index = np.array(range(reward_store3.shape[0]))
    for i in range(10):
        print(i, np.argmax(reward_store3[:, i]), 5 - np.max(reward_store3[:, i]))
        plt.plot(index, reward_store3[:, i])
        plt.show()
    # reward_store = np.clip(reward_store, 0, 2)
    # plt.plot(index, np.mean(reward_store1, axis=1), label='0.9')
    # plt.plot(index, np.max(reward_store2, axis=1), label='0.95')
    # plt.plot(index, np.mean(reward_store2, axis=1), label='0.95')
    # plt.plot(index, np.min(reward_store2, axis=1), label='0.95')
    plt.plot(index, np.max(reward_store3, axis=1), label='0.95')
    plt.plot(index, np.mean(reward_store3, axis=1), label='0.98')
    plt.plot(index, np.min(reward_store3, axis=1), label='0.95')
    # plt.plot(index, np.mean(reward_store4, axis=1), label='0.99')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
 