import time
from EnvUAV.env import YawControlEnv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from tqdm import tqdm


def printPID(env):
    x_param = [env.x_controller.P, env.x_controller.I, env.x_controller.D]
    y_param = [env.y_controller.P, env.y_controller.I, env.y_controller.D]
    z_param = [env.z_controller.P, env.z_controller.I, env.z_controller.D]
    attitude_param = [
        env.attitude_controller.P,
        env.attitude_controller.I,
        env.attitude_controller.D,
    ]
    print("x_controller: P:", x_param[0], " I:", x_param[1], " D:", x_param[2])
    print("y_controller: P:", y_param[0], " I:", y_param[1], " D:", y_param[2])
    print("z_controller: P:", z_param[0], " I:", z_param[1], " D:", z_param[2])
    print(
        "atitude_controller: P:",
        attitude_param[0],
        " I:",
        attitude_param[1],
        " D:",
        attitude_param[2],
    )


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
            roll.append(env.current_ori[0])
            pitch.append(env.current_ori[1])
            yaw.append(env.current_ori[2])
            roll_target.append(0)
            pitch_target.append(0)
            yaw_target.append(target[3])

    env.close()
    error_x = np.sum(np.abs(np.array(x) - np.array(x_target)))
    error_y = np.sum(np.abs(np.array(y) - np.array(y_target)))
    error_z = np.sum(np.abs(np.array(z) - np.array(z_target)))
    error_pitch = np.sum(np.abs(np.array(pitch) - np.array(pitch_target)))
    error_roll = np.sum(np.abs(np.array(roll) - np.array(roll_target)))
    error_yaw = np.sum(np.abs(np.array(yaw) - np.array(yaw_target)))

    error_total = error_x + error_y + error_z + error_pitch + error_roll + error_yaw
    return error_total


def optimize():
    space = [(50.0, 60.0), (0.1, 10.0)]
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


def main():
    path = os.path.dirname(os.path.realpath(__file__))

    env = YawControlEnv()
    # env.attitude_controller.set_param(55, 1)

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
    main()
    # optimize()
