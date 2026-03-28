import time
from EnvUAV.env import YawControlEnv
from Agent import TD3
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/TD3_0.98/'
    if not os.path.exists(path):
        os.makedirs(path)
    agent = TD3(path, s_dim=2, gamma=0.99)
    agent.load_net('6', '95')
    # 6 95
    agent.var = 0
    env = YawControlEnv()

    x = []
    x_target = []
    roll = []

    y = []
    y_target = []
    pitch = []

    z = []
    z_target = []
    vel = []
    yaw = []

    action = []

    target = [5, -5]
    for episode in range(2):
        s = env.reset(target[episode])
        for ep_step in range(500):
            a = agent.get_action(s)
            s_, r, done, info = env.step(a[0])
            s = s_
            print(ep_step, env.current_pos[2], env.current_vel[2], s, r)

            action.append(a[0])

            x.append(env.current_pos[0])
            x_target.append(0)
            roll.append(env.current_ang[0])

            y.append(env.current_pos[1])
            y_target.append(0)
            pitch.append(env.current_ang[1])

            z.append(env.current_pos[2])
            z_target.append(0)
            vel.append(env.current_vel[2])
            yaw.append(env.current_ang[2])

    index = np.array(range(len(x))) * 0.01
    zeros = np.zeros_like(index)
    roll = np.array(roll) / np.pi * 180
    pitch = np.array(pitch) / np.pi * 180
    yaw = np.array(yaw) / np.pi * 180
    plt.subplot(3, 2, 1)
    plt.plot(index, x, label='x')
    plt.plot(index, x_target, label='x_target')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(index, pitch, label='pitch')
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(index, y, label='y')
    plt.plot(index, y_target, label='y_target')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(index, roll, label='roll')
    plt.plot(index, zeros)
    plt.plot(index, vel)
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(index, z, label='z')
    plt.plot(index, np.ones_like(yaw) * target[0])
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(index, yaw, label='yaw')
    plt.plot(index, action)
    plt.plot(index, zeros)
    plt.legend()

    plt.show()

    z = np.array(z)
    print(z.shape)
    z_record = np.empty(shape=[2, 500])
    z_record[0, :] = z[0:500]
    z_record[1, :] = z[500:1000]
    np.save(path + 'test.npy', z_record)
    plt.plot(np.array(range(500)), z_record[0, :])
    plt.plot(np.array(range(500)), z_record[1, :])
    plt.show()


if __name__ == '__main__':
    main()

