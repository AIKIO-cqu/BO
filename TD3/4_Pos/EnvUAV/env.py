import os
import time

from .uav import UAV
from .surrounding import Surrounding
from .controller import Controller
import numpy as np
import pybullet as p


class YawControlEnv:
    def __init__(self,
                 model='cf2x',
                 render=False,
                 random=True,
                 time_step=0.01):
        '''
        :param model: The model/type of the uav.
        :param render: Whether to render the simulation process
        :param random: Whether to use random initialization setting
        :param time_step: time_steps
        '''
        self.render = render
        self.model = model
        self.random = random
        self.time_step = time_step
        self.path = os.path.dirname(os.path.realpath(__file__))

        self.client = None
        self.time = None
        self.surr = None
        self.current_pos = self.last_pos = None
        self.current_ang = self.last_ang = None
        self.current_vel = self.last_vel = None
        self.current_ang_vel = self.last_ang_vel = None
        self.target = None
        self.uav = None
        self.attitude_controller = Controller(path=self.path, prefix='Attitude')
        self.z_controller = Controller(path=self.path, prefix='Z')
        self.xy_controller = Controller(path=self.path, prefix='XY', s_dim=6)

    def close(self):
        p.disconnect(self.client)

    def reset(self, base_pos, base_ang):
        # 若已经存在上一组，则关闭之，开启下一组训练
        if p.isConnected():
            p.disconnect(self.client)
        self.client = p.connect(p. GUI if self.render else p.DIRECT)
        self.time = 0.
        # 构建场景
        self.surr = Surrounding(client=self.client,
                                time_step=self.time_step)
        base_pos = base_pos
        base_ori = base_ang
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ang = self.last_ang = np.array(base_ori)
        self.current_vel = self.last_vel = np.array([0., 0., 0.])
        self.current_ang_vel = self.last_ang_vel = np.array([0., 0., 0.])
        self.uav = UAV(path=self.path,
                       client=self.client,
                       time_step=self.time_step,
                       base_pos=base_pos,
                       base_ori=p.getQuaternionFromEuler(base_ori))

    def step(self, target):
        self.target = target
        x_s, y_s = self._get_xy_s()
        z_s = self._get_z_s()
        xa = self.xy_controller.get_action(x_s)
        ya = self.xy_controller.get_action(y_s)
        za = self.z_controller.get_action(z_s)

        fx = xa*self.uav.M*5
        fy = ya*self.uav.M*5
        fz = self.uav.M*(self.uav.G + 5*za)

        yaw = target[3]
        roll = np.arcsin((np.sin(yaw) * fx - np.cos(yaw) * fy) / np.linalg.norm([fx, fy, fz]))
        pitch = np.arctan((np.cos(yaw) * fx + np.sin(yaw) * fy) / fz)
        f = fz / np.cos(self.current_ang[0]) / np.cos(self.current_ang[1])

        # roll = a*np.pi/6

        s1, s2, s3 = self._get_attitude_s(np.array([roll, pitch, yaw]))
        tau1 = self.attitude_controller.get_action(s1)
        tau2 = self.attitude_controller.get_action(s2)
        tau3 = self.attitude_controller.get_action(s3)

        self.uav.apply_action(f, tau1, tau2, tau3, self.time)
        p.stepSimulation()
        self.time += self.time_step

        self.last_pos = self.current_pos
        self.last_ang = self.current_ang
        self.last_vel = self.current_vel
        self.last_ang_vel = self.current_ang_vel

        current_pos, current_ang = p.getBasePositionAndOrientation(self.uav.id)
        current_ang = p.getEulerFromQuaternion(current_ang)
        current_vel, current_ang_vel = p.getBaseVelocity(self.uav.id)
        # 在环境当中，我们均以np.array的形式来存储。
        self.current_pos = np.array(current_pos)
        self.current_ang = np.array(current_ang)
        self.current_vel = np.array(current_vel)
        self.current_ang_vel = np.array(current_ang_vel)

    def _get_xy_s(self):
        ex = self.current_pos[0] - self.target[0]
        vx = self.current_vel[0]
        ey = self.current_pos[1] - self.target[1]
        vy = self.current_vel[1]
        e_h = self.current_pos[2] - self.target[2]
        v_h = self.current_vel[2]

        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.current_ang)), [3, 3])
        roll_ = np.arctan(R[1, 2]/R[2, 2])
        pitch_ = np.arctan(R[0, 2]/R[2, 2])
        roll_v = self.current_ang_vel[0]
        pitch_v = self.current_ang_vel[1]

        sx = np.array([ex, vx, np.sign(ex)*e_h, np.sign(ex)*v_h, pitch_, pitch_v])/3
        sy = np.array([ey, vy, np.sign(ey)*e_h, np.sign(ey)*v_h, roll_, -roll_v])/3
        return sx, sy

    def _get_attitude_s(self, target):
        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.current_ang)), [3, 3])
        R_d = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(target)), [3, 3])
        e_R = (np.matmul(R_d.T, R) - np.matmul(R.T, R_d)) / 2
        e = [e_R[1,2], e_R[2,0], e_R[0, 1]]
        v = np.matmul(self.current_ang_vel, R)
        s1, s2, s3 = [e[0], v[0]], [e[1], v[1]], [e[2], v[2]]
        return s1, s2, s3

    def _get_z_s(self):
        e = self.current_pos[2] - self.target[2]
        v = self.current_vel[2]
        s = [e, v]
        return s
