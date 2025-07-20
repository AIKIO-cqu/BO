import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import YawControlEnv
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import animation_Trajectory, printPID


# 生成目标轨迹
def generate_traj(shape_type, length):
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
    target_trajectory = np.vstack([tx, ty, tz, tpsi]).T

    return target_trajectory, name


# 测试 PID 参数
def test_fixed_traj(shape_type=0, length=5000, params=None):
    env = YawControlEnv()
    if params is not None:
        env.set_pid_params(PID_params=params)

    pos = []
    ang = []

    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    targets, name = generate_traj(shape_type, length)
    print(f"============PID: {name}============")

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
    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
    print("pos_error", np.mean(pos_error), np.std(pos_error))
    print("ang_error", np.mean(ang_error), np.std(ang_error))
    # print("error_total", np.mean(pos_error) + np.mean(ang_error))

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
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], label="target")
    ax.view_init(azim=45.0, elev=30)
    plt.legend()
    plt.show()

    index = np.array(range(length)) * 0.01
    zeros = np.zeros_like(index)
    plt.subplot(4, 2, 1)
    plt.plot(index, px, label="x")
    plt.plot(index, targets[:, 0], label="x_target")
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(index, pitch, label="pitch")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 3)
    plt.plot(index, py, label="y")
    plt.plot(index, targets[:, 1], label="y_target")
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(index, roll, label="roll")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 5)
    plt.plot(index, pz, label="z")
    plt.plot(index, targets[:, 2], label="z_target")
    plt.legend()

    plt.subplot(4, 2, 6)
    plt.plot(index, yaw, label="yaw")
    plt.plot(index, targets[:, 3], label="yaw_target")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 7)
    plt.plot(index, pos_error, label="pos_error")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 8)
    plt.plot(index, ang_error, label="ang_error")
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
        x_traget=targets[:, 0],
        y_traget=targets[:, 1],
        z_traget=targets[:, 2],
    )

    return pos_error, ang_error


# 测试 PID 参数 (2阶段)
def test_fixed_traj_2stage(shape_type=0, length=5000, first_length=300, params1=None, params2=None):
    env = YawControlEnv()
    pos = []
    ang = []
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    targets, name = generate_traj(shape_type, length)
    print(f"============PID: {name}============")

    if params1 is not None:
        env.set_pid_params(PID_params=params1)
    print("params of stage1:")
    printPID(env)
    for i in range(first_length):
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())

    if params2 is not None:
        env.set_pid_params(PID_params=params2)
    print("params of stage2:")
    printPID(env)
    for i in range(first_length, length):
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())

    env.close()

    # 将 pos 和 ang 列表转换为 NumPy 数组
    pos = np.array(pos)
    ang = np.array(ang)

    # 位置误差、角度误差
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
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], label="target")
    ax.view_init(azim=45.0, elev=30)
    plt.legend()
    plt.show()

    index = np.array(range(length)) * 0.01
    zeros = np.zeros_like(index)
    plt.subplot(4, 2, 1)
    plt.plot(index, px, label="x")
    plt.plot(index, targets[:, 0], label="x_target")
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(index, pitch, label="pitch")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 3)
    plt.plot(index, py, label="y")
    plt.plot(index, targets[:, 1], label="y_target")
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(index, roll, label="roll")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 5)
    plt.plot(index, pz, label="z")
    plt.plot(index, targets[:, 2], label="z_target")
    plt.legend()

    plt.subplot(4, 2, 6)
    plt.plot(index, yaw, label="yaw")
    plt.plot(index, targets[:, 3], label="yaw_target")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 7)
    plt.plot(index, pos_error, label="pos_error")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 8)
    plt.plot(index, ang_error, label="ang_error")
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
        x_traget=targets[:, 0],
        y_traget=targets[:, 1],
        z_traget=targets[:, 2],
    )

    return pos_error, ang_error


# 根据航点生成随机轨迹
def generate_random_trajectory(num_waypoints, position_range, length, draw=False):
    # 生成随机航点
    x_min, x_max = position_range[0]
    y_min, y_max = position_range[1]
    z_min, z_max = position_range[2]
    waypoints = np.zeros((num_waypoints, 3))
    waypoints[0] = np.array([0, 0, 0])  # 第一个航点固定为 (0, 0, 0)
    waypoints[1:, 0] = np.random.uniform(
        x_min, x_max, num_waypoints - 1
    )  # 随机生成 x 坐标
    waypoints[1:, 1] = np.random.uniform(
        y_min, y_max, num_waypoints - 1
    )  # 随机生成 y 坐标
    waypoints[1:, 2] = np.random.uniform(
        z_min, z_max, num_waypoints - 1
    )  # 随机生成 z 坐标

    # 计算时间轴
    dt = 0.01
    sim_time = length * dt
    t = np.linspace(0, sim_time, length)  # 仿真全局时间轴
    t_waypoints = np.linspace(0, sim_time, len(waypoints))  # 航点对应时间

    # 分别对 x, y, z 进行三次样条插值
    waypoints = np.array(waypoints)
    cs_x = CubicSpline(t_waypoints, waypoints[:, 0])  # x 插值
    cs_y = CubicSpline(t_waypoints, waypoints[:, 1])  # y 插值
    cs_z = CubicSpline(t_waypoints, waypoints[:, 2])  # z 插值

    # 生成平滑轨迹
    x = cs_x(t)
    y = cs_y(t)
    z = cs_z(t)

    # 计算速度 (vx, vy) 并根据速度方向计算偏航角 psi
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    psi = np.arctan2(vy, vx)

    # 平滑处理，确保轨迹连续
    x = gaussian_filter1d(x, sigma=5)
    y = gaussian_filter1d(y, sigma=5)
    z = gaussian_filter1d(z, sigma=5)
    psi = gaussian_filter1d(psi, sigma=5)

    # 组合轨迹
    trajectory = np.vstack((x, y, z, psi)).T

    # 画图
    if draw:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        # 第一个点
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            trajectory[0, 2],
            c="g",
            marker="o",
            label="Start Point",
        )
        # 第二个到导数第二个点
        for i in range(1, len(waypoints) - 1):
            ax.scatter(
                waypoints[i, 0], waypoints[i, 1], waypoints[i, 2], c="C0", marker="o"
            )
        # 最后一个点
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            trajectory[-1, 2],
            c="r",
            marker="o",
            label="End Point",
        )
        ax.view_init(azim=45.0, elev=30)
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(t, x, label="x")
        plt.plot(t, y, label="y")
        plt.plot(t, z, label="z")
        plt.legend()
        plt.show()

    return trajectory, waypoints