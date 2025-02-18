import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from EnvUAV.env_BO import YawControlEnv
import torch
import torch.nn as nn
import torch.optim as optim
from utils import generate_target_trajectory, test_fixed_traj

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


# 定义神经网络
class PIDNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=1):
        super(PIDNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, X):
        self.eval()  # 设置模型为评估模式
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(
                next(self.parameters()).device
            )
            y_pred = self.forward(X_tensor)
        return y_pred.cpu().numpy()


def incremental_update_pid(
    targets, model, initial_pid, bounds, step_size=100, ang_weight=0.5
):
    """
    增量学习: 在整个轨迹过程中, 每step_size步优化一次PID参数
    """
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    current_pid = initial_pid
    best_pid = initial_pid
    best_error = float("inf")

    for i in range(0, len(targets), step_size):
        target_segment = targets[i : i + step_size]  # 取当前 step_size 范围内的目标点

        env.new_PD_params(PD_params=current_pid)  # 设置当前 PID 参数
        pos, ang = [], []

        for target in target_segment:
            env.step(target)
            pos.append(env.current_pos.tolist())
            ang.append(env.current_ori.tolist())

        pos = np.array(pos)
        ang = np.array(ang)

        # 计算误差
        pos_error = np.sqrt(np.sum((pos - target_segment[:, :3]) ** 2, axis=1))
        ang_error = np.degrees(np.abs((ang[:, 2] - target_segment[:, 3])))
        current_error = np.mean(pos_error) + np.mean(ang_error) * ang_weight

        print(f"Step {i}/{len(targets)}, Error: {current_error:.4f}")

        # 生成新候选 PID 参数
        X_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (10, 8))
        y_preds = model.predict(X_candidates)

        # 选择误差最小的参数
        X_next = X_candidates[np.argmin(y_preds)]

        # 真实测试新参数
        y_next = simulate_trajectory_multi_control(targets[i : i + step_size], X_next)

        # 只在误差降低时更新 PID
        if y_next < current_error:
            env.new_PD_params(PD_params=X_next)
            current_pid = X_next
            best_error = y_next
            print(f"Updated PID parameters: {current_pid}")

    env.close()
    return best_pid


def train_pid_net_incrementally(model, optimizer, X_new, y_new, X_history, y_history):
    """
    增量训练神经网络，防止遗忘
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    # 合并新数据与部分旧数据
    replay_ratio = 0.3
    replay_size = int(replay_ratio * len(X_history))
    X_train = np.vstack((X_new, X_history[:replay_size]))
    y_train = np.vstack((y_new, y_history[:replay_size]))

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = nn.MSELoss()(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    return loss.item()


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

    # 初始化神经网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PIDNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 初始 PID 采样
    n_initial_samples = 10
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial_samples, 8))
    y_sample = np.array(
        [simulate_trajectory_multi_control(target_trajectory, X) for X in X_sample]
    ).reshape(-1, 1)

    # 训练初始模型
    for _ in range(200):
        train_pid_net_incrementally(
            model, optimizer, X_sample, y_sample, X_sample, y_sample
        )

    # 增量优化 PID
    best_pid = incremental_update_pid(
        target_trajectory,
        model,
        initial_pid=X_sample[np.argmin(y_sample)],
        bounds=bounds,
    )

    test_fixed_traj(best_pid, 5000, 0)
    test_fixed_traj(best_pid, 5000, 1)
    test_fixed_traj(best_pid, 5000, 2)
