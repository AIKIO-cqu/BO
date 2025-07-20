import os
from .uav import UAV
from .surrounding import Surrounding
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

    def close(self):
        p.disconnect(self.client)

    def reset(self, target=None):
        # 若已经存在上一组，则关闭之，开启下一组训练
        if p.isConnected():
            p.disconnect(self.client)
        self.client = p.connect(p. GUI if self.render else p.DIRECT)
        self.time = 0.
        # 构建场景
        self.surr = Surrounding(client=self.client,
                                time_step=self.time_step)
        # 初始化时便最好用float
        base_pos = np.array([0., 0., 0.])
        if target is None:
            base_ori = np.array([0., 0., (np.random.rand()-0.5)*np.pi])
            self.target = np.array([0, 0, 0])
        else:
            base_ori = np.array([0., 0., target])
            self.target = np.array([0, 0, 0])
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ang = self.last_ang = np.array(base_ori)
        self.current_vel = self.last_vel = np.array([0., 0., 0.])
        self.current_ang_vel = self.last_ang_vel = np.array([0., 0., 0.])
        self.uav = UAV(path=self.path,
                       client=self.client,
                       time_step=self.time_step,
                       base_pos=base_pos,
                       base_ori=p.getQuaternionFromEuler(base_ori))
        return self._get_s()

    def step(self, a):
        self.uav.apply_action(a, self.time)
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

        s_ = self._get_s()
        r = self._get_r()
        done = abs(self.current_ang[2]) > np.pi/2
        infor = None
        return s_, r, done, infor

    def _get_s(self):
        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.current_ang)), [3, 3])
        R_d = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.target)), [3, 3])
        e_R = (np.matmul(R_d.T, R) - np.matmul(R.T, R_d)) / 2
        e = e_R[0, 1] # x:[1,2], y[2, 0], z[0,1]
        vel = np.matmul(self.current_ang_vel, R)
        v = vel[2]
        s = [e, v]
        return s

    def _get_r(self):
        last_R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.last_ang)), [3, 3])
        current_R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.current_ang)), [3, 3])
        R_d = np.reshape(p.getMatrixFromQuaternion((p.getQuaternionFromEuler(self.target))), [3, 3])
        last_e_R = (np.matmul(R_d.T, last_R) - np.matmul(last_R.T, R_d)) / 2
        current_e_R = (np.matmul(R_d.T, current_R) - np.matmul(current_R.T, R_d)) / 2
        last_e = last_e_R[0, 1]
        current_e = current_e_R[0, 1]
        r = (abs(last_e) - abs(current_e))
        return r

    # def _get_r(self):
    #     last_y = self.last_ang[2]
    #     current_y = self.current_ang[2]
    #     target = self.target[2]
    #     last_diff = _get_diff(last_y, target)
    #     current_diff = _get_diff(current_y, target)
    #     r = (abs(last_diff) - abs(current_diff))
    #     return r


def _get_diff(ang, target):
    diff = (target - ang + np.pi) % (np.pi*2) - np.pi
    return diff
