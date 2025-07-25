from EnvUAV.env import YawControlEnv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import animation_Trajectory, printPID


def main():
    shape_type = 0
    path = os.path.dirname(os.path.realpath(__file__))

    env = YawControlEnv()

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
    elif shape_type == 2:
        name = "Spiral"  # 半径先增大后减小的螺旋形状
        index = np.array(range(length)) / length * 4  # 轨迹参数，4 圈
        radius = 2 + 0.3 * np.sin(np.pi * index)  # 半径先增大后减小
        tx = radius * np.cos(1.5 * np.pi * index)  # x 方向的螺旋
        ty = radius * np.sin(1.5 * np.pi * index)  # y 方向的螺旋
        tz = 0.5 * index - 1  # z 方向逐渐上升
        tpsi = np.cos(2 * np.pi * index) * np.pi / 4  # 偏航角周期变化
    else:
        raise ValueError("shape_type must be 0, 1, or 2")

    targets = np.vstack([tx, ty, tz, tpsi]).T

    for i in range(length):
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()

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

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    px = pos[:, 0]
    py = pos[:, 1]
    pz = pos[:, 2]
    roll = ang[:, 0]
    pitch = ang[:, 1]
    yaw = ang[:, 2]

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
    main()
