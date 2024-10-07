import time
from EnvUAV.env import YawControlEnv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from animation import animation_Trajectory
from bayes_opt import BayesianOptimization


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


def objective(P, D):
    env = YawControlEnv()
    env.attitude_controller.set_param(P, D)

    pos = []
    ang = []

    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    length = 5000

    name = "Four-leaf clover"  # 四叶草形状
    index = np.array(range(length)) / length * 2
    tx = 2 * np.sin(2 * np.pi * index) * np.cos(np.pi * index)
    ty = 2 * np.sin(2 * np.pi * index) * np.sin(np.pi * index)
    tz = -np.sin(2 * np.pi * index) * np.cos(np.pi * index) - np.sin(
        2 * np.pi * index
    ) * np.sin(np.pi * index)
    tpsi = np.sin(4 * np.pi * index) * np.pi / 4 * 3

    # name = "Ellipse"  # 圆形
    # index = np.array(range(length)) / length
    # tx = 2 * np.cos(2 * np.pi * index)
    # ty = 2 * np.sin(2 * np.pi * index)
    # tz = -np.cos(2 * np.pi * index) - np.sin(2 * np.pi * index)
    # tpsi = np.sin(2 * np.pi * index) * np.pi / 3 * 2

    targets = np.vstack([tx, ty, tz, tpsi]).T
    for i in range(length):
        target = targets[i, :]
        env.step(target)

        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()

    pos = np.array(pos)
    ang = np.array(ang)

    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.sqrt(np.sum((ang - targets[:, 3:]) ** 2, axis=1))
    error_total = np.mean(pos_error) + np.mean(ang_error)
    return -error_total


def optimize():
    pbounds = {"P": (55, 70), "D": (0.1, 2)}
    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=1)
    optimizer.maximize(n_iter=1000)
    print(optimizer.max)
    return optimizer.max


def main(result):
    path = os.path.dirname(os.path.realpath(__file__))

    env = YawControlEnv()
    env.attitude_controller.set_param(result["params"]["P"], result["params"]["D"])

    pos = []
    ang = []

    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    length = 5000
    name = "Four-leaf clover"  # 四叶草形状
    index = np.array(range(length)) / length * 2
    tx = 2 * np.sin(2 * np.pi * index) * np.cos(np.pi * index)
    ty = 2 * np.sin(2 * np.pi * index) * np.sin(np.pi * index)
    tz = -np.sin(2 * np.pi * index) * np.cos(np.pi * index) - np.sin(
        2 * np.pi * index
    ) * np.sin(np.pi * index)
    tpsi = np.sin(4 * np.pi * index) * np.pi / 4 * 3

    # name = "Ellipse"  # 圆形
    # index = np.array(range(length)) / length
    # tx = 2 * np.cos(2 * np.pi * index)
    # ty = 2 * np.sin(2 * np.pi * index)
    # tz = -np.cos(2 * np.pi * index) - np.sin(2 * np.pi * index)
    # tpsi = np.sin(2 * np.pi * index) * np.pi / 3 * 2

    targets = np.vstack([tx, ty, tz, tpsi]).T

    # np.save(path + "/PID_" + name + "_targets.npy", targets)

    start_time = time.time()
    for i in range(length):
        target = targets[i, :]
        env.step(target)

        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()
    end_time = time.time()
    total_time = end_time - start_time
    print("total_time", total_time)  # 总计算时间

    # 打印PID参数
    printPID(env)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    position = np.array(pos)
    px = position[:, 0]
    py = position[:, 1]
    pz = position[:, 2]
    attitude = np.array(ang)
    roll = attitude[:, 0]
    pitch = attitude[:, 1]
    yaw = attitude[:, 2]

    # np.save(file=path+'/Trajectory/quad_uav.npy', arr=position)
    # np.save(file=path+'/Trajectory/quad_target.npy', arr=targets)
    # np.save(file=path+'/Trajectory/circle_uav.npy', arr=position)
    # np.save(file=path+'/Trajectory/circle_target.npy', arr=targets)

    # 画图
    ax.plot(px, py, pz, label="track")
    ax.plot(tx, ty, tz, label="target")
    ax.view_init(azim=45.0, elev=30)
    plt.legend()
    plt.show()

    index = np.array(range(length)) * 0.01
    zeros = np.zeros_like(index)
    plt.subplot(3, 2, 1)
    plt.plot(index, px, label="x")
    plt.plot(index, tx, label="x_target")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(index, pitch, label="pitch")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(index, py, label="y")
    plt.plot(index, ty, label="y_target")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(index, roll, label="roll")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(index, pz, label="z")
    plt.plot(index, tz, label="z_target")
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(index, yaw, label="yaw")
    plt.plot(index, tpsi, label="yaw_target")
    plt.plot(index, zeros)
    plt.legend()

    plt.show()

    pos = np.array(pos)
    ang = np.array(ang)
    # np.save(path + "/PID_" + name + "_pos.npy", pos)
    # np.save(path + "/PID_" + name + "_ang.npy", ang)

    print("PID ", name)
    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.sqrt(np.sum((ang - targets[:, 3:]) ** 2, axis=1))
    print("pos_error", np.mean(pos_error), np.std(pos_error))
    print("ang_error", np.mean(ang_error), np.std(ang_error))
    print("total_error", np.mean(pos_error) + np.mean(ang_error))

    # 动画
    animation_Trajectory(
        t_all=np.array(range(length)) * 0.01,
        dt=0.01,
        x_list=pos[:, 0],
        y_list=pos[:, 1],
        z_list=pos[:, 2],
        x_traget=tx,
        y_traget=ty,
        z_traget=tz,
    )


if __name__ == "__main__":
    result = optimize()
    main(result)
