import time
from EnvUAV.env import YawControlEnv
from Agent import TD3
import os
import json
import argparse


def main(gamma, index):
    # start = time.time()
    # start_time = time.time()
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/TD3_'+str(gamma)+'/'+str(index)
    if not os.path.exists(path):
        os.makedirs(path)
    agent = TD3(path, s_dim=2, gamma=gamma)
    env = YawControlEnv()

    step = 0
    for episode in range(8):
        s = env.reset()
        for ep_step in range(64):
            a = agent.get_action(s)
            s_, r, done, info = env.step(a[0])
            if done:
                episode -= 1
                break
            agent.store_transition(s, a, s_, r)
            s = s_
            step += 1
            if step >= 512:
                break
        if step >= 512:
            break

    for episode in range(200):
        s = env.reset()
        init_error = s[0]
        for ep_step in range(500):
            a = agent.get_action(s)
            s_, r, done, info = env.step(a[0])
            if done:
                break
            agent.store_transition(s, a, s_, r)
            s = s_
        last_error = s[0]
        print('episode: ', episode,
              ' init_error: ', round(init_error, 3),
              ' last_error: ', round(last_error, 5),
              ' variance: ', agent.var)
        agent.store_net(str(episode))
        if done:
            episode -= 1
    env.close()


if __name__ == '__main__':
    for i in [2]:#range(10):
        main(0.98, i)

