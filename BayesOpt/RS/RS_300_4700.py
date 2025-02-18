import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from EnvUAV.env_BO import YawControlEnv
from utils import test_fixed_traj_300_4700, generate_target_trajectory


# 模拟多控制器轨迹跟踪任务（前300步）
def simulate_trajectory_first_300_steps(targets, pid_params, ang_weight=0.5):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    env.new_PD_params(PD_params=pid_params)
    pos = []
    ang = []
    for i in range(300):  # 只计算前300步
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()
    pos = np.array(pos)
    ang = np.array(ang)

    # 计算误差
    pos_error = np.sqrt(np.sum((pos - targets[:300, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[:300, 3])))

    # 返回加权误差
    return np.mean(pos_error) + np.mean(ang_error) * ang_weight


# 模拟多控制器轨迹跟踪任务（后4700步）
def simulate_trajectory_last_4700_steps(
    targets, pid_params, X_best_stage1, ang_weight=0.3
):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    pos = []
    ang = []

    # 阶段1：前300步使用 X_best_stage1
    env.new_PD_params(PD_params=X_best_stage1)
    for i in range(300):
        target = targets[i, :]
        env.step(target)

    # 阶段2：后4700步使用阶段2优化参数 pid_params
    env.new_PD_params(PD_params=pid_params)
    for i in range(300, len(targets)):  # 计算后4700步
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()

    pos = np.array(pos)
    ang = np.array(ang)

    # 计算误差
    pos_error = np.sqrt(np.sum((pos - targets[300:5000, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[300:5000, 3])))

    # 返回加权误差
    return np.mean(pos_error) + np.mean(ang_error) * ang_weight


if __name__ == "__main__":
    # 定义优化边界
    bounds = np.array(
        [
            [0.0, 5.0],
            [0.0, 2.0],
            [0.0, 5.0],
            [0.0, 2.0],
            [15.0, 30.0],
            [5.0, 15.0],
            [10.0, 30.0],
            [0.0, 5.0],
        ]
    )

    # 目标轨迹
    shape_type_traj = 2  # 0: Circle, 1: Four-leaf clover, 2: Spiral
    length_traj = 5000  # 轨迹长度
    target_trajectory, name_traj = generate_target_trajectory(
        shape_type_traj, length_traj
    )

    # 分阶段优化
    # 阶段 1：前300步
    print("Optimizing for first 300 steps...")
    best_params_1 = None
    best_error_1 = float("inf")  # 初始化最小误差为正无穷
    n_iterations = 100
    for iteration in range(n_iterations):
        # 随机生成一个参数组合
        random_params_1 = np.array(
            [np.random.uniform(low, high) for low, high in bounds]
        )
        # 计算目标函数的值
        error_1 = simulate_trajectory_first_300_steps(
            target_trajectory, random_params_1
        )
        # 如果当前的误差比最好的误差小，更新最好的参数和误差
        if error_1 < best_error_1:
            best_error_1 = error_1
            best_params_1 = random_params_1
        # 输出当前进度
        print(
            f"Iteration {iteration + 1}/{n_iterations}, Error: {error_1:.4f}, Best Error: {best_error_1:.4f}"
        )
    # 获取最优参数
    X_best_stage1 = best_params_1
    print(f"Best PID parameters for first 300 steps: {X_best_stage1}")

    # 阶段 2：后4700步
    print("Optimizing for remaining 4700 steps...")
    best_params_2 = None
    best_error_2 = float("inf")  # 初始化最小误差为正无穷
    n_iterations = 100
    for iteration in range(n_iterations):
        # 随机生成一个参数组合
        random_params_2 = np.array(
            [np.random.uniform(low, high) for low, high in bounds]
        )
        # 计算目标函数的值
        error_2 = simulate_trajectory_last_4700_steps(
            target_trajectory, random_params_2, X_best_stage1
        )
        # 如果当前的误差比最好的误差小，更新最好的参数和误差
        if error_2 < best_error_2:
            best_error_2 = error_2
            best_params_2 = random_params_2
        # 输出当前进度
        print(
            f"Iteration {iteration + 1}/{n_iterations}, Error: {error_2:.4f}, Best Error: {best_error_2:.4f}"
        )
    # 获取最优参数
    X_best_stage2 = best_params_2
    print(f"Best PID parameters for remaining 4700 steps: {X_best_stage2}")

    test_fixed_traj_300_4700(
        X_best_stage1, X_best_stage2, length=length_traj, shape_type=0
    )
    test_fixed_traj_300_4700(
        X_best_stage1, X_best_stage2, length=length_traj, shape_type=1
    )
    test_fixed_traj_300_4700(
        X_best_stage1, X_best_stage2, length=length_traj, shape_type=2
    )
