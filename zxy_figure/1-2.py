import numpy as np
import matplotlib.pyplot as plt


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


def draw_Ellipse():
    shape_type = 0
    length = 5000
    targets, name = generate_target_trajectory(shape_type, length)

    pos_PID = np.load("D:\Project\BO\zxy_figure\pos_PID_Ellipse.npy")
    pos_LQR = np.load("D:\Project\BO\zxy_figure\pos_LQR_Ellipse.npy")
    pos_MPC = np.load("D:\Project\BO\zxy_figure\pos_MPC_Ellipse.npy")
    pos_TD3 = np.load("D:\Project\BO\zxy_figure\pos_TD3_Ellipse.npy")
    pos_RS = np.load("D:\Project\BO\zxy_figure\pos_PID+RS_Ellipse.npy")
    pos_BO = np.load("D:\Project\BO\zxy_figure\pos_PID+BO_Ellipse.npy")
    pos_HBO = np.load("D:\Project\BO\zxy_figure\pos_PID+HBO_Ellipse.npy")

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos_PID[:, 0], pos_PID[:, 1], pos_PID[:, 2], label="PID")
    ax.plot(pos_LQR[:, 0], pos_LQR[:, 1], pos_LQR[:, 2], label="LQR")
    ax.plot(pos_MPC[:, 0], pos_MPC[:, 1], pos_MPC[:, 2], label="MPC")
    ax.plot(pos_TD3[:, 0], pos_TD3[:, 1], pos_TD3[:, 2], label="TD3")
    ax.plot(pos_RS[:, 0], pos_RS[:, 1], pos_RS[:, 2], label="RS")
    ax.plot(pos_BO[:, 0], pos_BO[:, 1], pos_BO[:, 2], label="BO")
    ax.plot(pos_HBO[:, 0], pos_HBO[:, 1], pos_HBO[:, 2], label="HBO")

    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], label="target", color='black')
    ax.view_init(azim=45.0, elev=30)
    plt.legend()
    plt.show()


def draw_Four():
    shape_type = 1
    length = 5000
    targets, name = generate_target_trajectory(shape_type, length)

    pos_PID = np.load("D:\Project\BO\zxy_figure\pos_PID_Four.npy")
    pos_LQR = np.load("D:\Project\BO\zxy_figure\pos_LQR_Four.npy")
    pos_MPC = np.load("D:\Project\BO\zxy_figure\pos_MPC_Four.npy")
    pos_TD3 = np.load("D:\Project\BO\zxy_figure\pos_TD3_Four.npy")
    pos_RS = np.load("D:\Project\BO\zxy_figure\pos_PID+RS_Four.npy")
    pos_BO = np.load("D:\Project\BO\zxy_figure\pos_PID+BO_Four.npy")
    pos_HBO = np.load("D:\Project\BO\zxy_figure\pos_PID+HBO_Four.npy")

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos_PID[:, 0], pos_PID[:, 1], pos_PID[:, 2], label="PID")
    ax.plot(pos_LQR[:, 0], pos_LQR[:, 1], pos_LQR[:, 2], label="LQR")
    ax.plot(pos_MPC[:, 0], pos_MPC[:, 1], pos_MPC[:, 2], label="MPC")
    ax.plot(pos_TD3[:, 0], pos_TD3[:, 1], pos_TD3[:, 2], label="TD3")
    ax.plot(pos_RS[:, 0], pos_RS[:, 1], pos_RS[:, 2], label="RS")
    ax.plot(pos_BO[:, 0], pos_BO[:, 1], pos_BO[:, 2], label="BO")
    ax.plot(pos_HBO[:, 0], pos_HBO[:, 1], pos_HBO[:, 2], label="HBO")

    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], label="target", color='black')
    ax.view_init(azim=45.0, elev=30)
    plt.legend()
    plt.show()


def draw_Spiral():
    shape_type = 2
    length = 5000
    targets, name = generate_target_trajectory(shape_type, length)

    pos_PID = np.load("D:\Project\BO\zxy_figure\pos_PID_Spiral.npy")
    pos_LQR = np.load("D:\Project\BO\zxy_figure\pos_LQR_Spiral.npy")
    pos_MPC = np.load("D:\Project\BO\zxy_figure\pos_MPC_Spiral.npy")
    pos_TD3 = np.load("D:\Project\BO\zxy_figure\pos_TD3_Spiral.npy")
    pos_RS = np.load("D:\Project\BO\zxy_figure\pos_PID+RS_Spiral.npy")
    pos_BO = np.load("D:\Project\BO\zxy_figure\pos_PID+BO_Spiral.npy")
    pos_HBO = np.load("D:\Project\BO\zxy_figure\pos_PID+HBO_Spiral.npy")

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos_PID[:, 0], pos_PID[:, 1], pos_PID[:, 2], label="PID")
    ax.plot(pos_LQR[:, 0], pos_LQR[:, 1], pos_LQR[:, 2], label="LQR")
    ax.plot(pos_MPC[:, 0], pos_MPC[:, 1], pos_MPC[:, 2], label="MPC")
    ax.plot(pos_TD3[:, 0], pos_TD3[:, 1], pos_TD3[:, 2], label="TD3")
    ax.plot(pos_RS[:, 0], pos_RS[:, 1], pos_RS[:, 2], label="RS")
    ax.plot(pos_BO[:, 0], pos_BO[:, 1], pos_BO[:, 2], label="BO")
    ax.plot(pos_HBO[:, 0], pos_HBO[:, 1], pos_HBO[:, 2], label="HBO")

    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], label="target", color='black')
    ax.view_init(azim=45.0, elev=30)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    draw_Ellipse()
    draw_Four()
    draw_Spiral()


# 测试用的参数
PID_params = [1, 0.77, 1, 0.77, 20, 10.5, 20, 3.324]
PID_RS_params = [
    2.074697139425429,
    1.8002557547010822,
    3.2318102503303674,
    1.6096471309988258,
    22.124103912645428,
    12.188682900042139,
    17.760882764135705,
    2.559802781742307,
]
PID_BO_params = [
    1.4903032285039675,
    0.6555869336657205,
    3.7072932982417224,
    1.8422600068867023,
    20.62216163513675,
    13.64999389298014,
    19.4343066034699,
    2.62139093544266,
]
PID_HBO_params = [
    2.4072319861877944,
    1.1177837202417729,
    0.931257950917781,
    0.4007567734145976,
    15.18091718709083,
    5.013179923390205,
    27.713004830064556,
    3.1048918772723426,
]
