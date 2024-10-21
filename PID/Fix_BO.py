import time
from EnvUAV.env import YawControlEnv
import os
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from utils import (
    printPID,
    animation_Fix,
    calculate_peak,
    calculate_error,
    calculate_rise,
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
    return -np.mean(error_total)


def optimize():
    pbounds = {
        "Px": (0.0, 5.0),
        "Dx": (0.0, 5.0),
        "Py": (0.0, 5.0),
        "Dy": (0.0, 5.0),
        "Pz": (15.0, 30.0),
        "Dz": (5.0, 15.0),
        "Pa": (15.0, 30.0),
        "Da": (0.0, 5.0),
    }
    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=1)
    optimizer.maximize(n_iter=4000)
    print(optimizer.max)
    return optimizer.max


def main(result):
    path = os.path.dirname(os.path.realpath(__file__))

    env = YawControlEnv()
    env.x_controller.set_param(result["params"]["Px"], result["params"]["Dx"])
    env.y_controller.set_param(result["params"]["Py"], result["params"]["Dy"])
    env.z_controller.set_param(result["params"]["Pz"], result["params"]["Dz"])
    env.attitude_controller.set_param(result["params"]["Pa"], result["params"]["Da"])

    pos = []
    ang = []
    x = []
    y = []
    z = []
    x_target = []
    y_target = []
    z_target = []
    pitch = []
    roll = []
    yaw = []
    pitch_target = []
    roll_target = []
    yaw_target = []

    env.reset(base_pos=np.array([5, -5, 2]), base_ori=np.array([0, 0, 0]))
    targets = np.array([[0, 0, 0, np.pi / 3]])

    start_time = time.time()

    for episode in range(len(targets)):
        target = targets[episode, :]
        for ep_step in range(500):
            env.step(target)

            pos.append(env.current_pos.tolist())
            ang.append(env.current_ori.tolist())
            x.append(env.current_pos[0])
            y.append(env.current_pos[1])
            z.append(env.current_pos[2])
            x_target.append(target[0])
            y_target.append(target[1])
            z_target.append(target[2])
            roll.append(env.current_ori[0])
            pitch.append(env.current_ori[1])
            yaw.append(env.current_ori[2])
            roll_target.append(0)
            pitch_target.append(0)
            yaw_target.append(target[3])

    env.close()
    end_time = time.time()
    total_time = end_time - start_time
    print("total_time", total_time)  # 总计算时间

    # 打印PID参数
    printPID(env)

    # 位置误差、角度误差
    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.sqrt(np.sum((ang - targets[:, 3:]) ** 2, axis=1))
    error_total = np.mean(pos_error) + np.mean(ang_error)
    print("pos_error", np.mean(pos_error), np.std(pos_error))
    print("ang_error", np.mean(ang_error), np.std(ang_error))
    print("error_total", error_total)  # 平均总误差
    print("=====================================")

    # 计算峰值误差、误差和上升时间
    trace = np.concatenate(
        [
            np.array(x).reshape(-1, 1),
            np.array(y).reshape(-1, 1),
            np.array(z).reshape(-1, 1),
            np.array(yaw).reshape(-1, 1),
        ],
        axis=1,
    )
    peak = [calculate_peak(trace[:, i], target[i]) for i in range(4)]
    error = [calculate_error(trace[:, i], target[i]) for i in range(4)]
    rise = [calculate_rise(trace[:, i], target[i]) for i in range(4)]
    print("peak", peak)
    print("error", error)
    print("rise", rise)
    print("=====================================")

    # 画图
    index = np.array(range(len(x))) * 0.01
    zeros = np.zeros_like(index)

    roll = np.array(roll) / np.pi * 180
    pitch = np.array(pitch) / np.pi * 180
    yaw = np.array(yaw) / np.pi * 180
    roll_target = np.array(roll_target) / np.pi * 180
    pitch_target = np.array(pitch_target) / np.pi * 180
    yaw_target = np.array(yaw_target) / np.pi * 180

    plt.subplot(3, 2, 1)
    plt.plot(index, x, label="x")
    plt.plot(index, x_target, label="x_target")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(index, pitch, label="pitch")
    plt.plot(index, pitch_target, label="pitch_target")
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(index, y, label="y")
    plt.plot(index, y_target, label="y_target")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(index, roll, label="roll")
    plt.plot(index, roll_target, label="roll_target")
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(index, z, label="z")
    plt.plot(index, z_target, label="z_target")
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(index, yaw, label="yaw")
    plt.plot(index, yaw_target, label="yaw_target")
    plt.legend()

    plt.show()

    pos = np.array(pos)
    ang = np.array(ang)
    # np.save(path + "/PID_" + name + "_pos.npy", pos)
    # np.save(path + "/PID_" + name + "_ang.npy", ang)

    # 动画
    animation_Fix(
        t_all=np.array(range(len(x))) * 0.01,
        dt=0.01,
        x_list=x,
        y_list=y,
        z_list=z,
        x_traget=targets[:, 0],
        y_traget=targets[:, 1],
        z_traget=targets[:, 2],
    )


if __name__ == "__main__":
    result = optimize()
    main(result)
