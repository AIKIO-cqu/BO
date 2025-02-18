import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from EnvUAV.env_BO import YawControlEnv
from utils import generate_target_trajectory
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


ang_weight = 0.3


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
    return np.mean(pos_error) + np.mean(ang_error) * ang_weight


class IncrementalPIDOptimizer:
    def __init__(self, bounds, init_pids):
        self.bounds = bounds
        self.gpr = GaussianProcessRegressor(
            kernel=C(1.0) * RBF(1.0) + WhiteKernel(), normalize_y=True
        )
        self.X = []
        self.y = []
        self.pid_1, self.pid_2 = init_pids

    def generate_candidate(self):
        if len(self.X) < 3:  # 初始阶段随机采样
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        else:
            X_test = np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=(50, 8)
            )
            mu = self.gpr.predict(X_test)
            return X_test[np.argmin(mu)]


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
    shape_type_traj = 0  # 0: Circle, 1: Four-leaf clover, 2: Spiral
    length_traj = 5000  # 轨迹长度
    target_trajectory, name_traj = generate_target_trajectory(
        shape_type_traj, length_traj
    )

    pid_1 = [3.4014, 1.9846, 4.4171, 1.0579, 23.5734, 11.8431, 15.2957, 1.9989]
    pid_2 = [0.7907, 0.3808, 3.8757, 0.7896, 24.8834, 10.5618, 27.3392, 0.9489]

    optimizer = IncrementalPIDOptimizer(bounds, (pid_1, pid_2))
    current_pid = pid_1.copy()
    best_errors = []

    for chunk_start in range(0, 5000, 100):
        chunk = target_trajectory[chunk_start : chunk_start + 100]
        base_pid = pid_1 if chunk_start < 300 else pid_2

        # 生成候选参数并评估
        pid_new = optimizer.generate_candidate()
        error_new = simulate_trajectory_multi_control(chunk, pid_new)
        error_base = simulate_trajectory_multi_control(chunk, base_pid)

        # 增量学习更新
        optimizer.X.append(pid_new)
        optimizer.y.append(error_new)
        if len(optimizer.X) > 2:
            optimizer.gpr.fit(optimizer.X, optimizer.y)

        # 选择最优参数
        if error_new < error_base:
            current_pid = pid_new
            best_errors.append(error_new)
        else:
            current_pid = base_pid
            best_errors.append(error_base)

        print(
            f"Step {chunk_start}-{chunk_start+100}: "
            f"Error {min(error_new, error_base):.2f}"
        )
