import time
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


# 使用随机搜索算法优化PID参数
def optimize():
    best_error = 3.5
    best_param = {
        "best_Px": 0,
        "best_Dx": 0,
        "best_Py": 0,
        "best_Dy": 0,
        "best_Pz": 0,
        "best_Dz": 0,
        "best_Pa": 0,
        "best_Da": 0,
    }
    for i in range(4000):
        print(i)
        Px = np.random.uniform(0, 5)
        Dx = np.random.uniform(0, 5)
        Py = np.random.uniform(0, 5)
        Dy = np.random.uniform(0, 5)
        Pz = np.random.uniform(15, 30)
        Dz = np.random.uniform(5, 15)
        Pa = np.random.uniform(15, 30)
        Da = np.random.uniform(0, 5)
        error = objective(Px, Dx, Py, Dy, Pz, Dz, Pa, Da)
        if error < best_error:
            print(i, "error: ", error, "params: ", Px, Dx, Py, Dy, Pz, Dz, Pa, Da)
            best_error = error
            best_param["best_Px"] = Px
            best_param["best_Dx"] = Dx
            best_param["best_Py"] = Py
            best_param["best_Dy"] = Dy
            best_param["best_Pz"] = Pz
            best_param["best_Dz"] = Dz
            best_param["best_Pa"] = Pa
            best_param["best_Da"] = Da
    print("best_error: ", best_error)
    print("best_param: ", best_param)
    return best_param


def main(result):
    path = os.path.dirname(os.path.realpath(__file__))
    env = YawControlEnv()

    # # 1000次迭代后的最优参数 平均总误差
    # env.x_controller.set_param(2.508935239227194, 2.347125280326763)
    # env.y_controller.set_param(1.970629641779923, 1.7580108721167143)
    # env.z_controller.set_param(23.792342973589367, 11.004785796237298)
    # env.attitude_controller.set_param(22.45551224390249, 3.883245039437792)

    env.x_controller.set_param(result["best_Px"], result["best_Dx"])
    env.y_controller.set_param(result["best_Py"], result["best_Dy"])
    env.z_controller.set_param(result["best_Pz"], result["best_Dz"])
    env.attitude_controller.set_param(result["best_Pa"], result["best_Da"])

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

    name = "Fixed1"
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
