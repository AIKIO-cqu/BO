import os
from .uav import UAV
from .surrounding import Surrounding
from .controller import AttitudePID, PositionPID
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
        self.current_ori = self.last_ori = None
        self.current_matrix = self.last_matrix = None
        self.current_vel = self.last_vel = None
        self.current_ang_vel = self.last_ang_vel = None
        self.target = None
        self.uav = None

        self.x_controller = PositionPID(P=1.732, I=0, D=1.366)
        self.y_controller = PositionPID(P=1.732, I=0, D=1.366)
        self.z_controller = PositionPID(P=2.58, I=0, D=1.42)
        self.attitude_controller = AttitudePID(P=10, I=0, D=1.41)

    def close(self):
        p.disconnect(self.client)

    def reset(self, base_pos, base_ori):
        # 若已经存在上一组，则关闭之，开启下一组训练
        if p.isConnected():
            p.disconnect(self.client)
        self.client = p.connect(p. GUI if self.render else p.DIRECT)
        self.time = 0.
        # 构建场景
        self.surr = Surrounding(client=self.client,
                                time_step=self.time_step)
        # 初始化时便最好用float
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ori = self.last_ori = np.array(base_ori)
        self.current_matrix = self.last_matrix = np.array([[1., 0., 0.],
                                                           [0., 1., 0.],
                                                           [0., 0., 1.]])
        self.current_vel = self.last_vel = np.array([0., 0., 0.])
        self.current_ang_vel = self.last_ang_vel = np.array([0., 0., 0.])
        self.target = np.zeros(3)
        self.uav = UAV(path=self.path,
                       client=self.client,
                       time_step=self.time_step,
                       base_pos=base_pos,
                       base_ori=p.getQuaternionFromEuler(base_ori))

        self.x_controller.reset()
        self.y_controller.reset()
        self.z_controller.reset()
        self.attitude_controller.reset()
        return self._get_s()

    def step(self, target):
        x_a = self.x_controller.computControl(self.current_pos[0], target[0])
        y_a = self.y_controller.computControl(self.current_pos[1], target[1])
        z_a = self.z_controller.computControl(self.current_pos[2], target[2])

        fx = self.uav.M*5*x_a
        fy = self.uav.M*5*y_a
        fz = self.uav.M*(self.uav.G+5*z_a)

        yaw = target[3]
        roll = np.arcsin((np.sin(yaw) * fx - np.cos(yaw) * fy) / np.linalg.norm([fx, fy, fz]))
        pitch = np.arctan((np.cos(yaw) * fx + np.sin(yaw) * fy) / fz)

        # print(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw])))
        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw])), [3, 3])
        # f = np.dot([fx, fy, fz], R[:, 2])

        # f = fz / np.cos(self.current_ori[0]) / np.cos(self.current_ori[1])
        f = fz / np.cos(roll) / np.cos(pitch)
        tau = self.attitude_controller.computControl(self.current_ori, [roll, pitch, yaw], self.current_ang_vel)

        tau_roll = 20*self.uav.J_xx*tau[0]
        tau_pitch = 20*self.uav.J_yy*tau[1]
        tau_yaw = 20*self.uav.J_zz*tau[2]

        self.uav.apply_action(np.array([f, tau_roll, tau_pitch, tau_yaw]), self.time)
        p.stepSimulation()
        self.time += self.time_step

        self.last_pos = self.current_pos
        self.last_ori = self.current_ori
        self.last_vel = self.current_vel
        self.last_ang_vel = self.current_ang_vel

        current_pos, current_ori = p.getBasePositionAndOrientation(self.uav.id)
        current_matrix = np.reshape(p.getMatrixFromQuaternion(current_ori), [3, 3])
        current_ori = p.getEulerFromQuaternion(current_ori)
        current_vel, current_ang_vel = p.getBaseVelocity(self.uav.id)
        # 在环境当中，我们均以np.array的形式来存储。
        self.current_pos = np.array(current_pos)
        self.current_ori = np.array(current_ori)
        self.current_matrix = current_matrix
        self.current_vel = np.array(current_vel)
        self.current_ang_vel = np.matmul(current_ang_vel, current_matrix)

        # self._check_collision()
        s_ = self._get_s()
        r = self._get_r()
        done = False
        infor = None
        return s_, r, done, infor

    def _get_s(self):
        return self._get_y_s(self.target[1])

    def _get_r(self):
        last_y = self.last_pos[1]
        current_y = self.current_pos[1]
        target = self.target[1]
        r = (abs(target - last_y) - abs(target - current_y))
        return r

    def _get_x_s(self, target):
        x = self.current_pos[0]
        x_v = self.current_vel[0]
        x_acc = (self.current_vel[0] - self.last_vel[0]) / self.time_step

        z = self.current_pos[2]
        z_v = self.current_vel[2]
        z_acc = (self.current_vel[2] - self.last_vel[2]) / self.time_step

        x_ang = self.current_matrix[0, 2]
        x_ang_v = (self.current_matrix[0, 2] - self.last_matrix[0, 2]) / self.time_step

        s = [target - x, x_v, x_acc, self.target[2]-z, z_v, z_acc, x_ang, x_ang_v]
        return s

    def _get_y_s(self, target):
        y = self.current_pos[1]
        y_v = self.current_vel[1]
        y_acc = (self.current_vel[1] - self.last_vel[1]) / self.time_step

        z = self.current_pos[2]
        z_v = self.current_vel[2]
        z_acc = (self.current_vel[2] - self.last_vel[2]) / self.time_step

        y_ang = self.current_matrix[1, 2]
        y_ang_v = (self.current_matrix[1, 2] - self.last_matrix[1, 2]) / self.time_step

        # s = [target - y, y_v, y_acc, self.target[2]-z, z_v, z_acc, y_ang, y_ang_v]
        s = [target - y, y_v, y_acc, y_ang, y_ang_v]
        return s

    def _get_z_s(self, target):
        z = self.current_pos[2]
        z_v = self.current_vel[2]
        z_acc = (self.current_vel[2] - self.last_vel[2]) / self.time_step
        target = target

        s = [target-z, z_v, z_acc]
        return s

    def _get_ang_s(self, target, dim):
        ang = self.current_ori[dim]
        ang_v = self.current_ang_vel[dim]
        ang_acc = (self.current_ang_vel[dim] - self.last_ang_vel[dim]) / self.time_step
        target = target
        diff = _get_diff(ang, target)
        s = [diff, ang_v, ang_acc]
        return s


def _get_diff(ang, target):
    diff = (target - ang + np.pi) % (np.pi*2) - np.pi
    return diff


