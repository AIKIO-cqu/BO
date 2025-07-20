import os
from .uav import UAV
from .surrounding import Surrounding
from .controller import AttitudePID, PositionPID, Controller
import numpy as np
import pybullet as p


class YawControlEnv:
    def __init__(self, model="cf2x", render=False, random=True, time_step=0.01):
        """
        :param model: The model/type of the uav.
        :param render: Whether to render the simulation process
        :param random: Whether to use random initialization setting
        :param time_step: time_steps
        """
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

        """PID_controller"""
        self.x_controller = PositionPID(P=1, I=0, D=0.77)
        self.y_controller = PositionPID(P=1, I=0, D=0.77)
        self.z_controller = PositionPID(P=20, I=0, D=10.5)
        self.attitude_controller = AttitudePID(P=20, I=0, D=3.324)

        """TD3_controller"""
        self.xy_controller_TD3 = Controller(path=self.path, prefix='XY', s_dim=6)
        self.z_controller_TD3 = Controller(path=self.path, prefix='Z')
        self.attitude_controller_TD3 = Controller(path=self.path, prefix='Attitude')

    def close(self):
        p.disconnect(self.client)
    
    def set_pid_params(self, PID_params):
        Px, Dx = PID_params[0], PID_params[1]
        Py, Dy = PID_params[2], PID_params[3]
        Pz, Dz = PID_params[4], PID_params[5]

        self.x_controller.set_param(Px, Dx)
        self.y_controller.set_param(Py, Dy)
        self.z_controller.set_param(Pz, Dz)

    def reset(self, base_pos, base_ori):
        # 若已经存在上一组，则关闭之，开启下一组训练
        if p.isConnected():
            p.disconnect(self.client)
        self.client = p.connect(p.GUI if self.render else p.DIRECT)
        self.time = 0.0
        # 构建场景
        self.surr = Surrounding(client=self.client, time_step=self.time_step)
        # 初始化时便最好用float
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ori = self.last_ori = np.array(base_ori)
        self.current_matrix = self.last_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.current_vel = self.last_vel = np.array([0.0, 0.0, 0.0])
        self.current_ang_vel = self.last_ang_vel = np.array([0.0, 0.0, 0.0])
        # self.target = np.zeros(3)
        self.uav = UAV(
            path=self.path,
            client=self.client,
            time_step=self.time_step,
            base_pos=base_pos,
            base_ori=p.getQuaternionFromEuler(base_ori),
        )

        self.x_controller.reset()
        self.y_controller.reset()
        self.z_controller.reset()
        self.attitude_controller.reset()

    def step(self, target):
        """PID_position"""
        x_a = self.x_controller.computControl(self.current_pos[0], target[0])
        y_a = self.y_controller.computControl(self.current_pos[1], target[1])
        z_a = self.z_controller.computControl(self.current_pos[2], target[2])
        fx = self.uav.M * 5 * x_a
        fy = self.uav.M * 5 * y_a
        fz = self.uav.M * (self.uav.G + 5 * z_a)

        """TD3_position"""
        # self.target = target
        # x_s, y_s = self._get_xy_s()
        # z_s = self._get_z_s()
        # x_a_TD3 = self.xy_controller_TD3.get_action(x_s)
        # y_a_TD3 = self.xy_controller_TD3.get_action(y_s)
        # z_a_TD3 = self.z_controller_TD3.get_action(z_s)
        # fx = self.uav.M * 5 * x_a_TD3
        # fy = self.uav.M * 5 * y_a_TD3
        # fz = self.uav.M * (self.uav.G + 5 * z_a_TD3)

        yaw = target[3]
        roll = np.arcsin((np.sin(yaw) * fx - np.cos(yaw) * fy) / np.linalg.norm([fx, fy, fz]))
        pitch = np.arctan((np.cos(yaw) * fx + np.sin(yaw) * fy) / fz)
        # f = fz / np.cos(roll) / np.cos(pitch)
        f = fz / np.cos(self.current_ori[0]) / np.cos(self.current_ori[1])  # TD3写法！！！

        """PID_attitude"""
        # tau = self.attitude_controller.computControl(
        #     self.current_ori, [roll, pitch, yaw], self.current_ang_vel
        # )
        # tau_roll = 20 * self.uav.J_xx * tau[0]
        # tau_pitch = 20 * self.uav.J_yy * tau[1]
        # tau_yaw = 20 * self.uav.J_zz * tau[2]

        """TD3_attitude"""
        s1, s2, s3 = self._get_attitude_s(np.array([roll, pitch, yaw]))
        tau_roll_TD3 = self.attitude_controller_TD3.get_action(s1)
        tau_pitch_TD3 = self.attitude_controller_TD3.get_action(s2)
        tau_yaw_TD3 = self.attitude_controller_TD3.get_action(s3)
        tau_TD3 = np.array([tau_roll_TD3, tau_pitch_TD3, tau_yaw_TD3])
        tau_roll = 20 * self.uav.J_xx * tau_TD3[0]
        tau_pitch = 20 * self.uav.J_yy * tau_TD3[1]
        tau_yaw = 20 * self.uav.J_zz * tau_TD3[2]

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
        # self.current_ang_vel = np.matmul(current_ang_vel, current_matrix)
        self.current_ang_vel = np.array(current_ang_vel)  # TD3写法！！！

    def _get_xy_s(self):
        ex = self.current_pos[0] - self.target[0]
        vx = self.current_vel[0]
        ey = self.current_pos[1] - self.target[1]
        vy = self.current_vel[1]
        e_h = self.current_pos[2] - self.target[2]
        v_h = self.current_vel[2]

        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.current_ori)), [3, 3])
        roll_ = np.arctan(R[1, 2]/R[2, 2])
        pitch_ = np.arctan(R[0, 2]/R[2, 2])
        roll_v = self.current_ang_vel[0]
        pitch_v = self.current_ang_vel[1]

        sx = np.array([ex, vx, np.sign(ex)*e_h, np.sign(ex)*v_h, pitch_, pitch_v])/3
        sy = np.array([ey, vy, np.sign(ey)*e_h, np.sign(ey)*v_h, roll_, -roll_v])/3
        return sx, sy
    
    def _get_z_s(self):
        e = self.current_pos[2] - self.target[2]
        v = self.current_vel[2]
        s = [e, v]
        return s
    
    def _get_attitude_s(self, target):
        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.current_ori)), [3, 3])
        R_d = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(target)), [3, 3])
        e_R = (np.matmul(R_d.T, R) - np.matmul(R.T, R_d)) / 2
        e = [e_R[1,2], e_R[2,0], e_R[0, 1]]
        v = np.matmul(self.current_ang_vel, R)
        s1, s2, s3 = [e[0], v[0]], [e[1], v[1]], [e[2], v[2]]
        return s1, s2, s3
