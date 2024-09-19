import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# LQR函数，用于计算最优增益矩阵K
def lqr(A, B, Q, R):
    # 求解Riccati方程
    P = la.solve_continuous_are(A, B, Q, R)
    
    # 计算LQR增益矩阵K
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    
    return K

# 无人机动力学模型，状态方程 x_dot = A * x + B * u
# 状态 [x, y, z, vx, vy, vz]，位置和速度
A = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])

B = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# 定义LQR的权重矩阵
Q = np.diag([100, 100, 100, 10, 10, 10])  # 状态权重矩阵
R = np.diag([1, 1, 1])                    # 控制权重矩阵

# 计算LQR增益矩阵
K = lqr(A, B, Q, R)

# 模拟无人机的运动，使用LQR控制目标跟踪
dt = 0.01  # 时间步长
time = np.arange(0, 10, dt)  # 模拟时间

# 初始状态 [x, y, z, vx, vy, vz]
x = np.array([0, 0, 0, 0, 0, 0])  # 无人机初始位置和速度
target = np.array([10, 10, 10, 0, 0, 0])  # 目标状态

# 记录状态和控制输入
state_history = []
control_history = []

for t in time:
    # 计算状态误差
    error = x - target
    
    # LQR控制输入 u = -K * error
    u = -np.dot(K, error)
    
    # 动力学更新: x_dot = A * x + B * u
    x_dot = np.dot(A, x) + np.dot(B, u)
    
    # 更新状态: x(t+1) = x(t) + x_dot * dt
    x = x + x_dot * dt
    
    # 记录状态和控制输入
    state_history.append(x[:3])  # 只记录位置 [x, y, z]
    control_history.append(u)

state_history = np.array(state_history)

# 绘制无人机的轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2], label='Trajectory')
ax.scatter(target[0], target[1], target[2], color='red', label='Target', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
