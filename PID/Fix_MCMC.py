import time
import scipy.stats
import seaborn as sns
from EnvUAV.env import YawControlEnv
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    printPID,
    calculate_peak,
    calculate_error,
    calculate_rise,
    animation_Fix,
)


def objective(P, D):
    env = YawControlEnv()
    env.x_controller.set_param(P, D)

    pos = []
    ang = []

    env.reset(base_pos=np.array([5, -5, 2]), base_ori=np.array([0, 0, 0]))
    targets = np.array([[0, 0, 0, np.pi / 3]])

    for episode in range(len(targets)):
        target = targets[episode, :]
        for ep_step in range(500):
            env.step(target)

            pos.append(env.current_pos.tolist())
            ang.append(env.current_ori.tolist())

    env.close()

    pos = np.array(pos)
    ang = np.array(ang)

    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
    error_total = np.mean(pos_error) + np.mean(ang_error)
    return np.mean(error_total)


def metropolis_hastings(p, iter=1000):
    x, y = 0.0, 0.0
    error = p(x, y)
    print("baseline:", error)
    samples = np.zeros((iter, 3))

    for i in range(iter):
        x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
        error_star = p(x_star, y_star)

        # 使用指数函数增加接受概率
        if np.random.rand() < min(1, 0.5 * np.exp((error - error_star) / error)):
            print(i, x_star, y_star, p(x_star, y_star))
            x, y = x_star, y_star
            error = error_star
        samples[i] = np.array([x, y, error])

    return samples


if __name__ == "__main__":
    samples = metropolis_hastings(objective, 10000)
    np.save("samples.npy", samples)
    sns.jointplot(x=samples[:, 0], y=samples[:, 1])
    plt.show()

    # 绘图显示3个子图
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    axs[0].plot(samples[:, 0])
    axs[0].set_ylabel("P")
    axs[1].plot(samples[:, 1])
    axs[1].set_ylabel("D")
    axs[2].plot(samples[:, 2])
    axs[2].set_ylabel("Error")
    plt.show()
