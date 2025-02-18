import os
from .uav import UAV
from .surrounding import Surrounding
from .controller import AttitudePID, PositionPID
import numpy as np
import pybullet as p
from .windModel import Wind
import copy
from scipy.integrate import ode


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

        self.count_step = 0
        self.wind_model = Wind('SINE', 2.0, 90, -15)

        self.x_controller = PositionPID(P=1, I=0, D=0.77)
        self.y_controller = PositionPID(P=1, I=0, D=0.77)
        self.z_controller = PositionPID(P=20, I=0, D=10.5)
        self.attitude_controller = AttitudePID(P=20, I=0, D=3.324)

    def close(self):
        p.disconnect(self.client)

    def new_PD_params(self, PD_params):
        Px, Dx = PD_params[0], PD_params[1]
        Py, Dy = PD_params[2], PD_params[3]
        Pz, Dz = PD_params[4], PD_params[5]
        Pa, Da = PD_params[6], PD_params[7]

        self.x_controller.set_param(Px, Dx)
        self.y_controller.set_param(Py, Dy)
        self.z_controller.set_param(Pz, Dz)
        self.attitude_controller.set_param(Pa, Da)

    def is_params_equal(self, PD_params):
        if (
            self.x_controller.P == PD_params[0]
            and self.x_controller.D == PD_params[1]
            and self.y_controller.P == PD_params[2]
            and self.y_controller.D == PD_params[3]
            and self.z_controller.P == PD_params[4]
            and self.z_controller.D == PD_params[5]
            and self.attitude_controller.P == PD_params[6]
            and self.attitude_controller.D == PD_params[7]
        ):
            return True
        return False

    def reset(self, base_pos, base_ori):
        # # 若已经存在上一组，则关闭之，开启下一组训练
        # if p.isConnected():
        #     p.disconnect(self.client)
        self.client = p.connect(p.GUI if self.render else p.DIRECT)
        self.time = 0.0
        self.count_step = 0
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
        self.target = np.zeros(3)
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
        return self._get_s()

    def step(self, target):
        self.count_step += 1
        x_a = self.x_controller.computControl(self.current_pos[0], target[0])
        y_a = self.y_controller.computControl(self.current_pos[1], target[1])
        z_a = self.z_controller.computControl(self.current_pos[2], target[2])

        fx = self.uav.M * 5 * x_a
        fy = self.uav.M * 5 * y_a
        fz = self.uav.M * (self.uav.G + 5 * z_a)

        # # 添加风力扰动
        # [velW, qW1, qW2] = self.wind_model.randomWind(self.time)
        # fx += velW * np.cos(qW1) * np.cos(qW2) * 0.02
        # fy += velW * np.sin(qW1) * np.cos(qW2) * 0.02
        # fz += velW * np.sin(qW2) * 0.02

        yaw = target[3]
        roll = np.arcsin(
            (np.sin(yaw) * fx - np.cos(yaw) * fy) / np.linalg.norm([fx, fy, fz])
        )
        pitch = np.arctan((np.cos(yaw) * fx + np.sin(yaw) * fy) / fz)

        # print(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw])))
        R = np.reshape(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw])),
            [3, 3],
        )
        # f = np.dot([fx, fy, fz], R[:, 2])

        f = fz / np.cos(self.current_ori[0]) / np.cos(self.current_ori[1])
        f = fz / np.cos(roll) / np.cos(pitch)
        tau = self.attitude_controller.computControl(
            self.current_ori, [roll, pitch, yaw], self.current_ang_vel
        )

        tau_roll = 20 * self.uav.J_xx * tau[0]
        tau_pitch = 20 * self.uav.J_yy * tau[1]
        tau_yaw = 20 * self.uav.J_zz * tau[2]

        # # 添加风力扰动
        # tau_roll += velW * np.cos(qW1) * np.cos(qW2) * 0.001 * 0.0397
        # tau_pitch += velW * np.sin(qW1) * np.cos(qW2) * 0.001 * 0.0397
        # tau_yaw += velW * np.sin(qW2) * 0.001 * 0.0397

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
        r = abs(target - last_y) - abs(target - current_y)
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

        s = [target - x, x_v, x_acc, self.target[2] - z, z_v, z_acc, x_ang, x_ang_v]
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

        s = [target - z, z_v, z_acc]
        return s

    def _get_ang_s(self, target, dim):
        ang = self.current_ori[dim]
        ang_v = self.current_ang_vel[dim]
        ang_acc = (self.current_ang_vel[dim] - self.last_ang_vel[dim]) / self.time_step
        target = target
        diff = _get_diff(ang, target)
        s = [diff, ang_v, ang_acc]
        return s

    def get_state(self):
        """
        获取当前环境的状态，包括所有需要存储的变量。
        """
        state = {
            # 基础环境变量
            "count_step": self.count_step,
            "current_ang_vel": self.current_ang_vel.copy(),
            "current_matrix": self.current_matrix.copy(),
            "current_ori": self.current_ori.copy(),
            "current_pos": self.current_pos.copy(),
            "current_vel": self.current_vel.copy(),
            "last_ang_vel": self.last_ang_vel.copy(),
            "last_ori": self.last_ori.copy(),
            "last_pos": self.last_pos.copy(),
            "last_vel": self.last_vel.copy(),
            "time": self.time,
            # UAV状态
            "motor_speed": (
                self.uav.motor_speed.copy()
                if self.uav.motor_speed is not None
                else None
            ),
            "integrator_state": (
                {
                    "y": self.uav.integrator.y.copy(),
                    "t": self.uav.integrator.t,
                    "f_params": self.uav.integrator.f_params,
                }
                if self.uav.integrator is not None
                else None
            ),
            # 控制器状态
            "x_last_x": self.x_controller.last_x,
            "y_last_x": self.y_controller.last_x,
            "z_last_x": self.z_controller.last_x,
            # 保存 pybullet 的物理状态
            "pybullet_state": p.saveState(),
        }
        return state

    def set_state(self, state):
        """
        恢复环境的状态，包括所有需要恢复的变量。
        """
        # 恢复基础环境变量
        self.count_step = state["count_step"]
        self.current_ang_vel = state["current_ang_vel"].copy()
        self.current_matrix = state["current_matrix"].copy()
        self.current_ori = state["current_ori"].copy()
        self.current_pos = state["current_pos"].copy()
        self.current_vel = state["current_vel"].copy()
        self.last_ang_vel = state["last_ang_vel"].copy()
        self.last_ori = state["last_ori"].copy()
        self.last_pos = state["last_pos"].copy()
        self.last_vel = state["last_vel"].copy()
        self.time = state["time"]

        # 恢复 UAV 状态
        if state["motor_speed"] is not None:
            self.uav.motor_speed = state["motor_speed"].copy()
        # 恢复积分器状态
        if state["integrator_state"] is not None:
            self.uav.integrator = ode(self.uav.motor_dot).set_integrator(
                "dopri5", first_step="0.00005", atol="10e-6", rtol="10e-6"
            )
            self.uav.integrator.set_initial_value(
                state["integrator_state"]["y"], state["integrator_state"]["t"]
            )
            self.uav.integrator.set_f_params(state["integrator_state"]["f_params"])

        # 恢复控制器状态
        self.x_controller.last_x = state["x_last_x"]
        self.y_controller.last_x = state["y_last_x"]
        self.z_controller.last_x = state["z_last_x"]

        # 恢复 pybullet 的物理状态
        p.restoreState(state["pybullet_state"])


def _get_diff(ang, target):
    diff = (target - ang + np.pi) % (np.pi * 2) - np.pi
    return diff
