import time
from EnvUAV.env import YawControlEnv
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import animation_Trajectory, printPID
from bayes_opt import BayesianOptimization


def objective(Px, Dx, Py, Dy, Pz, Dz, Pa, Da):
    shape_type = 1
    env = YawControlEnv()
    env.x_controller.set_param(Px, Dx)
    env.y_controller.set_param(Py, Dy)
    env.z_controller.set_param(Pz, Dz)
    env.attitude_controller.set_param(Pa, Da)

    pos = []
    ang = []

    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    length = 5000
    if shape_type == 0:
        name = "Ellipse"  # 圆形
        index = np.array(range(length)) / length
        tx = 2 * np.cos(2 * np.pi * index)
        ty = 2 * np.sin(2 * np.pi * index)
        tz = -np.cos(2 * np.pi * index) - np.sin(2 * np.pi * index)
        tpsi = np.sin(2 * np.pi * index) * np.pi / 3 * 2

    elif shape_type == 1:
        name = "Four-leaf clover"  # 四叶草形状
        index = np.array(range(length)) / length * 2
        tx = 2 * np.sin(2 * np.pi * index) * np.cos(np.pi * index)
        ty = 2 * np.sin(2 * np.pi * index) * np.sin(np.pi * index)
        tz = -np.sin(2 * np.pi * index) * np.cos(np.pi * index) - np.sin(
            2 * np.pi * index
        ) * np.sin(np.pi * index)
        tpsi = np.sin(4 * np.pi * index) * np.pi / 4 * 3
    else:
        raise ValueError("shape_type must be 0 or 1")

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
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
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
    shape_type = 1
    path = os.path.dirname(os.path.realpath(__file__))

    env = YawControlEnv()

    # env.x_controller.set_param(5.0, 1.2927390502890024)
    # env.y_controller.set_param(5.0, 1.9068853449018406)
    # env.z_controller.set_param(23.316334370813053, 5.0)
    # env.attitude_controller.set_param(22.40380014127682, 5.0)

    env.x_controller.set_param(result["params"]["Px"], result["params"]["Dx"])
    env.y_controller.set_param(result["params"]["Py"], result["params"]["Dy"])
    env.z_controller.set_param(result["params"]["Pz"], result["params"]["Dz"])
    env.attitude_controller.set_param(result["params"]["Pa"], result["params"]["Da"])

    pos = []
    ang = []

    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    length = 5000
    if shape_type == 0:
        name = "Ellipse"  # 圆形
        index = np.array(range(length)) / length
        tx = 2 * np.cos(2 * np.pi * index)
        ty = 2 * np.sin(2 * np.pi * index)
        tz = -np.cos(2 * np.pi * index) - np.sin(2 * np.pi * index)
        tpsi = np.sin(2 * np.pi * index) * np.pi / 3 * 2

    elif shape_type == 1:
        name = "Four-leaf clover"  # 四叶草形状
        index = np.array(range(length)) / length * 2
        tx = 2 * np.sin(2 * np.pi * index) * np.cos(np.pi * index)
        ty = 2 * np.sin(2 * np.pi * index) * np.sin(np.pi * index)
        tz = -np.sin(2 * np.pi * index) * np.cos(np.pi * index) - np.sin(
            2 * np.pi * index
        ) * np.sin(np.pi * index)
        tpsi = np.sin(4 * np.pi * index) * np.pi / 4 * 3
    else:
        raise ValueError("shape_type must be 0 or 1")

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

    # 将 pos 和 ang 列表转换为 NumPy 数组
    pos = np.array(pos)
    ang = np.array(ang)

    # 位置误差、角度误差
    print("PID ", name)
    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
    print("pos_error", np.mean(pos_error), np.std(pos_error))
    print("ang_error", np.mean(ang_error), np.std(ang_error))
    print("error_total", np.mean(pos_error) + np.mean(ang_error))

    # 画图
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
