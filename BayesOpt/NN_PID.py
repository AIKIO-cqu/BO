import torch
import torch.nn as nn
from EnvUAV.env_BO import YawControlEnv
import numpy as np
from utils import generate_random_trajectory
import matplotlib.pyplot as plt


class PIDNetwork(nn.Module):
    def __init__(self, s_dim=6, t_dim=4):
        super(PIDNetwork, self).__init__()

        # Define the network layers
        self.fc1 = nn.Linear(s_dim + t_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 8)

    def forward(self, s, target):
        x = torch.cat((s, target), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        pid_params = self.fc3(x)
        return pid_params


def train(s_, target, optimizer):
    optimizer.zero_grad()
    pos_loss = torch.sqrt(torch.sum((s_[0:3] - target[0:3]) ** 2))
    ang_loss = torch.abs((s_[5] - target[3])) * (180.0 / torch.pi)
    loss = pos_loss + ang_loss * 0.1
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # Example training loop:
    target_trajectory, _ = generate_random_trajectory(
        5, [[-2, 2], [-2, 2], [-2, 2]], 5000
    )
    env = YawControlEnv()
    model = PIDNetwork(s_dim=6, t_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    for epoch in range(num_epochs):
        env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
        pos = []
        ang = []
        for i in range(len(target_trajectory)):
            target = target_trajectory[i, :]
            s = env.current_pos.tolist() + env.current_ori.tolist()
            pid_params = model(
                torch.tensor(s, dtype=torch.float32, requires_grad=True),
                torch.tensor(target, dtype=torch.float32, requires_grad=True),
            )
            env.new_PD_params(PD_params=pid_params.detach().numpy())
            env.step(target)
            s_ = env.current_pos.tolist() + env.current_ori.tolist()
            train(
                torch.tensor(s_, dtype=torch.float32, requires_grad=True),
                torch.tensor(target, dtype=torch.float32, requires_grad=True),
                optimizer,
            )
            pos.append(env.current_pos.tolist())
            ang.append(env.current_ori.tolist())

        pos = np.array(pos)
        ang = np.array(ang)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="track")
        ax.plot(
            target_trajectory[:, 0],
            target_trajectory[:, 1],
            target_trajectory[:, 2],
            label="target",
        )
        ax.view_init(azim=45.0, elev=30)
        plt.legend()
        plt.show()
