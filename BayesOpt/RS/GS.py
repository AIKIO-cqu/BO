import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from EnvUAV.env_BO import YawControlEnv
from utils import test_fixed_traj, generate_target_trajectory
from itertools import product


def get_bounds():
    # 定义每个参数的范围和分割数量
    param_ranges = [
        (0.0, 5.0, 2),  # Px
        (0.0, 5.0, 2),  # Dx
        (0.0, 5.0, 2),  # Py
        (0.0, 5.0, 2),  # Dy
        (10.0, 30.0, 4),  # Pz
        (5.0, 15.0, 2),  # Dz
        (10.0, 30.0, 4),  # Pa
        (0.0, 5.0, 2),  # Da
    ]

    # 计算每个参数的分割点
    splits = []
    for lower, upper, num_splits in param_ranges:
        step = (upper - lower) / num_splits
        split_points = [lower + i * step for i in range(num_splits)] + [upper]
        intervals = list(zip(split_points[:-1], split_points[1:]))
        splits.append(intervals)

    # 使用itertools.product来创建所有可能的组合
    combinations = list(product(*splits))

    return np.array(combinations)


# 模拟多控制器轨迹跟踪任务
def simulate_trajectory_multi_control(targets, pid_params):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    env.new_PD_params(PD_params=pid_params)
    pos = []
    ang = []
    for i in range(len(targets)):
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()
    pos = np.array(pos)
    ang = np.array(ang)
    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
    return np.mean(pos_error) + np.mean(ang_error) * 0.1
    # return np.mean(pos_error)


if __name__ == "__main__":
    # 目标轨迹
    shape_type_traj = 0  # 0: Circle, 1: Four-leaf clover, 2: Spiral
    length_traj = 5000  # 轨迹长度
    target_trajectory, name_traj = generate_target_trajectory(
        shape_type_traj, length_traj
    )

    best_params = None
    best_error = float("inf")  # 初始化最小误差为正无穷

    bounds_list = []
    error_list = []

    combinations = get_bounds()
    count = 0
    for bounds in combinations:
        print(f"第{count+1}次优化")
        
        # 随机生成一个参数组合
        random_params = np.array([np.random.uniform(low, high) for low, high in bounds])

        # 计算目标函数的值
        error = simulate_trajectory_multi_control(target_trajectory, random_params)

        bounds_list.append(bounds)
        error_list.append(error)

        # 如果当前的误差比最好的误差小，更新最好的参数和误差
        if error < best_error:
            best_error = error
            best_params = random_params

        count += 1

    # 获取最优参数
    X_best = best_params
    y_best = best_error

    print("Optimized trajectory shape:", name_traj, " sim_time:", 0.01 * length_traj)
    print(f"最优输入: {X_best}, 最优输出: {y_best}")

    test_fixed_traj(X_best, length=length_traj, shape_type=0)
    test_fixed_traj(X_best, length=length_traj, shape_type=1)
    test_fixed_traj(X_best, length=length_traj, shape_type=2)

    combined_list = list(zip(bounds_list, error_list))
    combined_list.sort(key=lambda x: x[1])
    bounds_list, error_list = zip(*combined_list)
    # 输出前10个最优参数
    for i in range(10):
        print(f"第{i+1}优参数: {bounds_list[i]}, 误差: {error_list[i]}")
