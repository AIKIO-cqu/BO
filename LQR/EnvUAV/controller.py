import torch
import torch.nn as nn
import numpy as np
import pybullet as p
import scipy


class PositionLQR:
    def __init__(self, A, B, Q, R):
        self.max_output = 1
        self.min_output = -1
        self.control_time_step = 0.01

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.last_x = 0

    def lqr(self):
        X = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
        K = np.matrix(scipy.linalg.inv(self.R) * (self.B.T * X))
        return K[0]

    def reset(self):
        self.last_x = 0

    def computControl(self, current_x, target):
        e = target - current_x
        vel_e = 0 - (current_x - self.last_x) / self.control_time_step
        self.last_x = current_x

        state_e = np.array([e, vel_e])
        k = self.lqr()
        output = np.dot(k, state_e)
        output = output[0, 0]
        return np.clip(output, self.min_output, self.max_output)


class AttitudeLQR:
    def __init__(self, A, B, Q, R):
        self.max_output = 1
        self.min_output = -1
        self.control_time_step = 0.01

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def lqr(self):
        X = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
        K = np.matrix(scipy.linalg.inv(self.R) * (self.B.T * X))
        return K[0]

    def reset(self):
        pass

    def computControl(self, ang, target, ang_vel):
        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ang)), [3, 3])
        R_d = np.reshape(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler(target)), [3, 3]
        )
        e_R = (np.matmul(R_d.T, R) - np.matmul(R.T, R_d)) / 2
        e = np.array([e_R[1, 2], e_R[2, 0], e_R[0, 1]])  # x:[1,2], y[2, 0], z[0,1]
        vel_e = ang_vel

        state_e = np.array([e, vel_e])
        k = self.lqr()
        output = np.dot(k, state_e)
        output = [output[0, 0], output[0, 1], output[0, 2]]
        return np.clip(output, self.min_output, self.max_output)
