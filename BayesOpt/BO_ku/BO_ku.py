import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from EnvUAV.env_BO import YawControlEnv
from utils import test_fixed_traj, generate_target_trajectory
from bayes_opt import BayesianOptimization


ang_weight = 0.3

# 目标轨迹
shape_type_traj = 0  # 0: Circle, 1: Four-leaf clover, 2: Spiral
length_traj = 5000  # 轨迹长度
target_trajectory, name_traj = generate_target_trajectory(shape_type_traj, length_traj)


# 模拟多控制器轨迹跟踪任务
def simulate_trajectory_multi_control(Px, Dx, Py, Dy, Pz, Dz, Pa, Da):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    env.new_PD_params(PD_params=[Px, Dx, Py, Dy, Pz, Dz, Pa, Da])
    pos = []
    ang = []
    for i in range(len(target_trajectory)):
        target = target_trajectory[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()
    pos = np.array(pos)
    ang = np.array(ang)
    pos_error = np.sqrt(np.sum((pos - target_trajectory[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - target_trajectory[:, 3])))
    return -(np.mean(pos_error) + np.mean(ang_error) * ang_weight)


if __name__ == "__main__":
    # 定义优化边界
    pounds = {
        "Px": (0.0, 5.0),
        "Dx": (0.0, 2.0),
        "Py": (0.0, 5.0),
        "Dy": (0.0, 2.0),
        "Pz": (10.0, 30.0),
        "Dz": (5.0, 15.0),
        "Pa": (10.0, 30.0),
        "Da": (0.0, 5.0),
    }

    # 定义优化器
    optimizer = BayesianOptimization(
        f=simulate_trajectory_multi_control,
        pbounds=pounds,
        verbose=2,  # 打印优化日志
        # random_state=42,
    )

    # 执行优化
    optimizer.maximize(
        init_points=20,  # 初始随机采样点的数量
        n_iter=100,  # 优化迭代次数
    )

    # 获取最优参数
    X_best = optimizer.max["params"]
    y_best = -optimizer.max["target"]  # 还原为最小化目标
    print(f"最优输入: {X_best}, 最优输出: {y_best}")

    print("=====================================")
    print("Optimized trajectory shape:", name_traj, " sim_time:", 0.01 * length_traj)

    params = [
        X_best["Px"],
        X_best["Dx"],
        X_best["Py"],
        X_best["Dy"],
        X_best["Pz"],
        X_best["Dz"],
        X_best["Pa"],
        X_best["Da"],
    ]

    test_fixed_traj(params, length=length_traj, shape_type=0)
    test_fixed_traj(params, length=length_traj, shape_type=1)
    test_fixed_traj(params, length=length_traj, shape_type=2)
