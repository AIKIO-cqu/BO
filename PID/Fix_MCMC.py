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


def objective(Px, Dx, Py, Dy, Pz, Dz, Pa, Da):
    env = YawControlEnv()
    env.x_controller.set_param(Px, Dx)
    env.y_controller.set_param(Py, Dy)
    env.z_controller.set_param(Pz, Dz)
    env.attitude_controller.set_param(Pa, Da)

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
    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.sqrt(np.sum((ang - targets[:, 3:]) ** 2, axis=1))
    error_total = np.mean(pos_error) + np.mean(ang_error)
    return np.mean(error_total)


def metropolis_hastings(P, chain):
    Px, Dx, Py, Dy, Pz, Dz, Pa, Da = 0, 0, 0, 0, 0, 0, 0, 0
    samples = []

    while True:
        Px_, Dx_, Py_, Dy_, Pz_, Dz_, Pa_, Da_ = (
            np.random.uniform(0, 5),
            np.random.uniform(0, 5),
            np.random.uniform(0, 5),
            np.random.uniform(0, 5),
            np.random.uniform(15, 30),
            np.random.uniform(5, 15),
            np.random.uniform(15, 30),
            np.random.uniform(0, 5),
        )

        p_moving = min(
            1,
            P(Px, Dx, Py, Dy, Pz, Dz, Pa, Da)
            / P(Px_, Dx_, Py_, Dy_, Pz_, Dz_, Pa_, Da_),
        )

        if scipy.stats.uniform.rvs() <= p_moving:
            samples.append([Px_, Dx_, Py_, Dy_, Pz_, Dz_, Pa_, Da_])
            Px, Dx, Py, Dy, Pz, Dz, Pa, Da = Px_, Dx_, Py_, Dy_, Pz_, Dz_, Pa_, Da_
        else:
            samples.append([Px, Dx, Py, Dy, Pz, Dz, Pa, Da])

        if len(samples) >= chain:
            break
    return np.array(samples)


if __name__ == "__main__":
    samples = metropolis_hastings(objective, 1000)
    print(samples)
