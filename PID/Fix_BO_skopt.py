import time
from EnvUAV.env import YawControlEnv
import os
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from tqdm import tqdm
from utils import printPID


def objective(params):
    env = YawControlEnv()
    env.attitude_controller.set_param(params[0], params[1])

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
    targets = np.array([[0, 0, 0, 0]])

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
            yaw.append(env.current_ori[2])
            yaw_target.append(target[3])

    env.close()
    error_x = np.array(x_target) - np.array(x)
    error_y = np.array(y_target) - np.array(y)
    error_z = np.array(z_target) - np.array(z)
    error_yaw = np.array(yaw_target) - np.array(yaw)
    error_total = (
        np.abs(error_x) + np.abs(error_y) + np.abs(error_z) + np.abs(error_yaw)
    )
    return -np.mean(error_total)


def optimize():
    space = [(0.1, 5.0), (0.1, 1.0)]
    n = 100

    # 创建一个 tqdm 进度条
    pbar = tqdm(total=n)

    # 定义一个回调函数来更新进度条
    def update_pbar(res):
        pbar.update()

    res = gp_minimize(
        objective, space, n_calls=n, callback=[update_pbar], random_state=0
    )

    pbar.close()

    print(res.x)
    return res.x


def main(result):
    path = os.path.dirname(os.path.realpath(__file__))

    env = YawControlEnv()
    env.attitude_controller.set_param(result[0], result[1])

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
    targets = np.array([[0, 0, 0, 0]])

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

    # 误差
    error_x = np.array(x_target) - np.array(x)
    error_y = np.array(y_target) - np.array(y)
    error_z = np.array(z_target) - np.array(z)
    error_yaw = np.array(yaw_target) - np.array(yaw)
    error_total = (
        np.abs(error_x) + np.abs(error_y) + np.abs(error_z) + np.abs(error_yaw)
    )
    print("error_x:", np.mean(np.abs(error_x)))
    print("error_y:", np.mean(np.abs(error_y)))
    print("error_z:", np.mean(np.abs(error_z)))
    print("error_yaw:", np.mean(np.abs(error_yaw)))
    print("error_total:", np.mean(error_total))  # 平均总误差

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


if __name__ == "__main__":
    result = optimize()
    main(result)
