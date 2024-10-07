import numpy as np


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
    print("=====================================")


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
    t2 = np.min(np.argwhere((x > target * 0.9)))
    return (t2 - t1) * 0.01
