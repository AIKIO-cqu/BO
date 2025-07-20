import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


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
    print("attitude_controller: P:", attitude_param[0], " I:", attitude_param[1], " D:", attitude_param[2])


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
    maxRange = (np.array([x_list.max() - x_list.min(),
                          y_list.max() - y_list.min(),
                          z_list.max() - z_list.min()]).max()* 0.5+ 0.5)
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
