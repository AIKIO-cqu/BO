import torch
import torch.nn as nn
from EnvUAV.env_BO import YawControlEnv
import numpy as np
from utils import generate_random_trajectory


class PIDNetwork(nn.Module):
    def __init__(self):
        super(PIDNetwork, self).__init__()

        # Define the network layers
        self.fc1 = nn.Linear(5000 * 4, 128)  # First hidden layer (input size 20000 -> 128)
        self.fc2 = nn.Linear(128, 128)  # Second hidden layer (128 -> 128)
        self.fc3 = nn.Linear(128, 8)  # Output layer (128 -> 8)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.contiguous().view(-1)  # 确保内存连续后再展平为一维向量
        x = torch.relu(self.fc1(x))  # Apply ReLU after first hidden layer
        x = torch.relu(self.fc2(x))  # Apply ReLU after second hidden layer
        x = self.fc3(x)  # Output layer
        return x


# 模拟多控制器轨迹跟踪任务
def simulate_trajectory_multi_control(targets, pid_params):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    pid_params = pid_params.detach().numpy()
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
    # ang = np.array(ang)
    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    # ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
    # return np.mean(pos_error) + np.mean(ang_error) * 0.1
    return torch.tensor(np.mean(pos_error), requires_grad=True)  # 确保返回 PyTorch 张量


# Example training loop:
target_trajectory, _ = generate_random_trajectory(5, [[-2, 2], [-2, 2], [-2, 2]], 5000)
model = PIDNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # 假设输入是目标轨迹数据，实际轨迹是由PID控制器控制得到的
    predicted_pid_params = model(target_trajectory)  # 网络输出PID参数
    loss = simulate_trajectory_multi_control(
        target_trajectory, predicted_pid_params
    )  # 使用PID控制器得到实际轨迹
    loss.backward()

    # 打印梯度值
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Epoch {epoch+1}, {name} grad: {param.grad.norm().item()}")

    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
