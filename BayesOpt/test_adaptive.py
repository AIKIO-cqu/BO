import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env_BO import YawControlEnv
from utils import generate_target_trajectory, animation_Trajectory, printPID

pid_params_list = np.load("D:\Project\BO\\final_params_adaptive_1_3.npy")

length = 5000
shape_type = 0
targets, name = generate_target_trajectory(shape_type, length)
print("PID ", name)

env = YawControlEnv()
env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

pos = []
ang = []

for i in range(length):
    target = targets[i, :]
    env.new_PD_params(PD_params=pid_params_list[i, :])
    env.step(target)
    pos.append(env.current_pos.tolist())
    ang.append(env.current_ori.tolist())

env.close()

# 将 pos 和 ang 列表转换为 NumPy 数组
pos = np.array(pos)
ang = np.array(ang)

# pos = np.load("/home/aikio/Projects/BO/pos_adaptive.npy")
# ang = np.load("/home/aikio/Projects/BO/ang_adaptive.npy")

# 位置误差、角度误差
pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
print("pos_error", np.mean(pos_error), np.std(pos_error))
print("ang_error", np.mean(ang_error), np.std(ang_error))
# print("error_total", np.mean(pos_error) + np.mean(ang_error))

# 画图
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

position = np.array(pos)
px = position[:, 0]
py = position[:, 1]
pz = position[:, 2]
attitude = np.array(ang)
roll = attitude[:, 0]
pitch = attitude[:, 1]
yaw = attitude[:, 2]

ax.plot(px, py, pz, label="track")
ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], label="target")
ax.view_init(azim=45.0, elev=30)
plt.legend()
plt.show()

index = np.array(range(length)) * 0.01
zeros = np.zeros_like(index)
plt.subplot(4, 2, 1)
plt.plot(index, px, label="x")
plt.plot(index, targets[:, 0], label="x_target")
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(index, pitch, label="pitch")
plt.plot(index, zeros)
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(index, py, label="y")
plt.plot(index, targets[:, 1], label="y_target")
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(index, roll, label="roll")
plt.plot(index, zeros)
plt.legend()

plt.subplot(4, 2, 5)
plt.plot(index, pz, label="z")
plt.plot(index, targets[:, 2], label="z_target")
plt.legend()

plt.subplot(4, 2, 6)
plt.plot(index, yaw, label="yaw")
plt.plot(index, targets[:, 3], label="yaw_target")
plt.plot(index, zeros)
plt.legend()

plt.subplot(4, 2, 7)
plt.plot(index, pos_error, label="pos_error")
plt.plot(index, zeros)
plt.legend()

plt.subplot(4, 2, 8)
plt.plot(index, ang_error, label="ang_error")
plt.plot(index, zeros)
plt.legend()

plt.show()

pos = np.array(pos)
ang = np.array(ang)

# 动画
animation_Trajectory(
    t_all=np.array(range(length)) * 0.01,
    dt=0.01,
    x_list=pos[:, 0],
    y_list=pos[:, 1],
    z_list=pos[:, 2],
    x_traget=targets[:, 0],
    y_traget=targets[:, 1],
    z_traget=targets[:, 2],
)
