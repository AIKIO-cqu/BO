import numpy as np
import pybullet as p


class PositionPID:
    def __init__(self, P, I, D):
        self.max_output = 1
        self.min_output = -1
        self.control_time_step = 0.01

        self.P = P
        self.I = I
        self.D = D
        self.last_x = 0
        self.integral_error = 0  # 积分误差初始化为0

    def reset(self):
        self.last_x = 0
        self.integral_error = 0  # 重置积分误差

    def set_param(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D

    def computControl(self, current_x, target):
        e = target - current_x
        self.integral_error += e * self.control_time_step  # 计算积分误差
        vel_e = 0 - (current_x - self.last_x) / self.control_time_step
        self.last_x = current_x
        output = self.P * e + self.D * vel_e
        return np.clip(output, -1, 1)


class AttitudePID:
    def __init__(self, P, I, D):
        self.max_output = 1
        self.min_output = -1
        self.control_time_step = 0.01

        self.P = P
        self.I = I
        self.D = D

        # 初始化积分误差
        self.integral_error = np.zeros(3)  # 误差为3维向量

    def reset(self):
        # 重置积分误差
        self.integral_error = np.zeros(3)

    def set_param(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D

    def computControl(self, ang, target, ang_vel):
        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ang)), [3, 3])
        R_d = np.reshape(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler(target)), [3, 3]
        )
        e_R = (np.matmul(R_d.T, R) - np.matmul(R.T, R_d)) / 2
        e = np.array([e_R[1, 2], e_R[2, 0], e_R[0, 1]])  # x:[1,2], y[2, 0], z[0,1]

        # 计算误差的积分部分
        self.integral_error += e * self.control_time_step
        # 限制积分误差避免积分饱和
        self.integral_error = np.clip(
            self.integral_error, self.min_output / self.I, self.max_output / self.I
        )

        vel_e = ang_vel
        output = self.P * e + self.I * self.integral_error - self.D * vel_e
        return np.clip(output, -1, 1)
