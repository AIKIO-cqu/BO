import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from EnvUAV.env_BO import YawControlEnv
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline


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


# 计算峰值 Peak value
def calculate_peak(x, target):
    target = np.abs(target - x[0])  # 目标距离
    x = np.abs(x - x[0])  # 已经移动的距离
    peak = np.max(x)
    return (peak - target) / target


# 计算 SSError
def calculate_error(x, target):
    x = np.array(x)[-50:]
    diff = np.abs(x - target)
    error = np.average(diff)
    return error


# 计算上升时间 Rise time
def calculate_rise(x, target):
    target = np.abs(target - x[0])  # 目标距离
    x = np.abs(x - x[0])  # 已经移动的距离
    t1 = np.max(np.argwhere((x < target * 0.1)))
    indices = np.argwhere((x > target * 0.9))
    if indices.size == 0:
        print("No value greater than 0.9")
        t2 = len(x)
    else:
        t2 = np.min(indices)
    return (t2 - t1) * 0.01


# 动画（定点跟踪）
def animation_Fix(t_all, dt, x_list, y_list, z_list, x_traget, y_traget, z_traget):
    numFrames = 4  # 每隔4帧绘制一次
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)

    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    (line,) = ax.plot([], [], [], "--", lw=1, color="blue")  # 四旋翼的飞行轨迹

    mid_x = (x_list.max() + x_list.min()) * 0.5
    mid_y = (y_list.max() + y_list.min()) * 0.5
    mid_z = (z_list.max() + z_list.min()) * 0.5
    maxRange = (
        np.array(
            [
                x_list.max() - x_list.min(),
                y_list.max() - y_list.min(),
                z_list.max() - z_list.min(),
            ]
        ).max()
        * 0.5
        + 0.5
    )
    ax.set_xlim3d([mid_x - maxRange, mid_x + maxRange])
    ax.set_xlabel("X")
    ax.set_ylim3d([mid_y - maxRange, mid_y + maxRange])
    ax.set_ylabel("Y")
    ax.set_zlim3d([mid_z - maxRange, mid_z + maxRange])
    ax.set_zlabel("Z")

    # 显示起始点和目标点
    ax.scatter(x_list[0], y_list[0], z_list[0], c="g", marker="o", label="Start Point")
    ax.scatter(x_traget, y_traget, z_traget, c="r", marker="o", label="Target Point")
    ax.legend(loc="upper right")

    # 显示时间
    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def updateLines(i):
        # 核心函数，定义了动画每一帧的更新逻辑
        time = t_all[i * numFrames]

        # 提取从模拟开始到当前帧的所有x、y、z坐标，用于绘制四旋翼的飞行轨迹
        x_from0 = x_list[0 : i * numFrames]
        y_from0 = y_list[0 : i * numFrames]
        z_from0 = z_list[0 : i * numFrames]

        line.set_data(x_from0, y_from0)
        line.set_3d_properties(z_from0)

        titleTime.set_text("Time = {:.2f} s".format(time))

    ani = animation.FuncAnimation(
        fig=fig,
        func=updateLines,
        frames=len(t_all[0:-1:numFrames]),  # 动画总帧数
        interval=dt * numFrames * 1000,  # 每帧间隔时间(ms)
        blit=False,
    )
    plt.show()
    return ani


# 动画（轨迹跟踪）
def animation_Trajectory(
    t_all, dt, x_list, y_list, z_list, x_traget, y_traget, z_traget
):
    numFrames = 8  # 每隔8帧绘制一次
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)

    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    (line,) = ax.plot(
        [], [], [], "--", lw=1, color="blue", label="uav"
    )  # 四旋翼的飞行轨迹

    mid_x = (x_list.max() + x_list.min()) * 0.5
    mid_y = (y_list.max() + y_list.min()) * 0.5
    mid_z = (z_list.max() + z_list.min()) * 0.5
    maxRange = (
        np.array(
            [
                x_list.max() - x_list.min(),
                y_list.max() - y_list.min(),
                z_list.max() - z_list.min(),
            ]
        ).max()
        * 0.5
        + 0.5
    )
    ax.set_xlim3d([mid_x - maxRange, mid_x + maxRange])
    ax.set_xlabel("X")
    ax.set_ylim3d([mid_y - maxRange, mid_y + maxRange])
    ax.set_ylabel("Y")
    ax.set_zlim3d([mid_z - maxRange, mid_z + maxRange])
    ax.set_zlabel("Z")

    # 绘制目标轨迹
    ax.plot(x_traget, y_traget, z_traget, "--", lw=1, color="green", label="target")
    ax.view_init(azim=45.0, elev=30)
    plt.legend()

    # 显示时间
    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def updateLines(i):
        # 核心函数，定义了动画每一帧的更新逻辑
        time = t_all[i * numFrames]

        # 提取从模拟开始到当前帧的所有x、y、z坐标，用于绘制四旋翼的飞行轨迹
        x_from0 = x_list[0 : i * numFrames]
        y_from0 = y_list[0 : i * numFrames]
        z_from0 = z_list[0 : i * numFrames]

        line.set_data(x_from0, y_from0)
        line.set_3d_properties(z_from0)

        titleTime.set_text("Time = {:.2f} s".format(time))

    ani = animation.FuncAnimation(
        fig=fig,
        func=updateLines,
        frames=len(t_all[0:-1:numFrames]),  # 动画总帧数
        interval=dt * numFrames * 1000,  # 每帧间隔时间(ms)
        blit=False,
    )
    plt.show()
    return ani


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


# 生成目标轨迹
def generate_target_trajectory(shape_type, length):
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
def test_params(params, shape_type, length):
    env = YawControlEnv()
    env.new_PD_params(PD_params=params)

    pos = []
    ang = []

    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    targets, name = generate_target_trajectory(shape_type, length)
    print("PID ", name)

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

    position = np.array(pos)
    px = position[:, 0]
    py = position[:, 1]
    pz = position[:, 2]
    attitude = np.array(ang)
    roll = attitude[:, 0]
    pitch = attitude[:, 1]
    yaw = attitude[:, 2]

    ax.plot(px, py, pz, label="track")
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], label="target")
    ax.view_init(azim=45.0, elev=30)
    plt.legend()
    plt.show()

    index = np.array(range(length)) * 0.01
    zeros = np.zeros_like(index)
    plt.subplot(3, 2, 1)
    plt.plot(index, px, label="x")
    plt.plot(index, targets[:, 0], label="x_target")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(index, pitch, label="pitch")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(index, py, label="y")
    plt.plot(index, targets[:, 1], label="y_target")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(index, roll, label="roll")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(index, pz, label="z")
    plt.plot(index, targets[:, 2], label="z_target")
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(index, yaw, label="yaw")
    plt.plot(index, targets[:, 3], label="yaw_target")
    plt.plot(index, zeros)
    plt.legend()

    plt.show()

    pos = np.array(pos)
    ang = np.array(ang)

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


if __name__ == "__main__":
    params = [4.748940157564426, 0.469647222284727, 4.992118011011425, 0.6634621009136538, 16.722161067485008, 5.699908009365622, 17.685060028970394, 0.47858791923934385]
    test_params(params, shape_type=0, length=5000)
    test_params(params, shape_type=1, length=5000)
    test_params(params, shape_type=2, length=5000)
