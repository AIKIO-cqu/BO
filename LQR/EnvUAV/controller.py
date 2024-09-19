import torch
import torch.nn as nn
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

    def reset(self):
        self.last_x = 0

    def computControl(self,
                      current_x,
                      target):
        e = target - current_x
        vel_e = 0 - (current_x - self.last_x)/ self.control_time_step
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

    def reset(self):
        pass

    def computControl(self,
                      ang,
                      target,
                      ang_vel):
        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ang)), [3, 3])
        R_d = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(target)), [3, 3])
        e_R = (np.matmul(R_d.T, R) - np.matmul(R.T, R_d))/2
        e = np.array([e_R[1,2], e_R[2,0], e_R[0,1]]) #x:[1,2], y[2, 0], z[0,1]
        vel_e = ang_vel
        output = self.P * e - self.D * vel_e
        return np.clip(output, -1, 1)
