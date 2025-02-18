import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from EnvUAV.env_BO import YawControlEnv
from utils import test_fixed_traj, generate_target_trajectory


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
    return np.mean(pos_error) + np.mean(ang_error) * 0.3
    # return np.mean(pos_error)


if __name__ == "__main__":
    # 定义优化边界
    bounds = np.array(
        [
            [0.0, 5.0],
            [0.0, 5.0],
            [0.0, 5.0],
            [0.0, 5.0],
            [15.0, 30.0],
            [5.0, 15.0],
            [10.0, 20.0],
            [0.0, 5.0],
        ]
    )

    # 目标轨迹
    shape_type_traj = 0  # 0: Circle, 1: Four-leaf clover, 2: Spiral
    length_traj = 5000  # 轨迹长度
    target_trajectory, name_traj = generate_target_trajectory(
        shape_type_traj, length_traj
    )

    best_params = None
    best_error = float("inf")  # 初始化最小误差为正无穷

    n_iterations = 100
    for iteration in range(n_iterations):
        # 随机生成一个参数组合
        random_params = np.array([np.random.uniform(low, high) for low, high in bounds])

        # 计算目标函数的值
        error = simulate_trajectory_multi_control(target_trajectory, random_params)

        # 如果当前的误差比最好的误差小，更新最好的参数和误差
        if error < best_error:
            best_error = error
            best_params = random_params

        # 输出当前进度
        print(
            f"Iteration {iteration + 1}/{n_iterations}, Error: {error:.4f}, Best Error: {best_error:.4f}"
        )

    # 获取最优参数
    X_best = best_params
    y_best = best_error

    print("=====================================")
    print("Optimized trajectory shape:", name_traj, " sim_time:", 0.01 * length_traj)
    print(f"最优输入: {X_best}, 最优输出: {y_best}")

    test_fixed_traj(X_best, shape_type=0, length=length_traj)
    test_fixed_traj(X_best, shape_type=1, length=length_traj)
    test_fixed_traj(X_best, shape_type=2, length=length_traj)
