import time
from EnvUAV.env import YawControlEnv
from Agent import TD3
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main(gamma):
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/TD3_'+str(gamma)+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    agent = TD3(path, s_dim=2, gamma=0.9)
    agent.var = 0
    env = YawControlEnv()

    reward_store = []

    for net_num in range(200):
        print(net_num)
        rewards = []
        for i in range(10):
            agent.load_net(prefix1=str(i), prefix2=str(net_num))
            #agent.load_net(prefix1=str(i), prefix2=str(net_num))
            reward = []
            s = env.reset(target=5)
            for ep_step in range(500):
                a = agent.get_action(s)
                s_, r, done, info = env.step(a[0])
                s = s_
                reward.append(r)
            rewards.append(np.sum(reward))
        reward_store.append(rewards)

    env.close()

    with open(path + '/disr'+str(gamma)+'.json', 'w') as f:
        json.dump(reward_store, f)

    with open(path + '/disr'+str(gamma)+'.json', 'r') as f:
        reward_store = json.load(f)

    reward_store = np.array(reward_store)
    # reward_store = np.clip(reward_store, 0, 2)
    mean = np.mean(reward_store, axis=1)
    std = np.std(reward_store, axis=1)

    index = np.array(range(mean.shape[0]))
    # plt.plot(index, np.clip(reward_store[:, 1], 0, 10))
    plt.plot(index, mean)
    plt.plot(index, np.min(reward_store, axis=1))
    plt.plot(index, np.max(reward_store, axis=1))
    plt.show()


if __name__ == '__main__':
    for gamma in [0.98]:#[0.9, 0.95, 0.98, 0.99]:
        main(gamma)
