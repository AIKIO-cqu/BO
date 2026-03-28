import pybullet as p
import yaml
import numpy as np
from scipy.integrate import ode


class UAV(object):
    def __init__(self,
                 path,
                 client,
                 time_step,
                 base_pos,
                 base_ori):
        self.path = path
        self.client = client
        with open(self.path+'/File/uva.yaml', 'r', encoding='utf-8') as F:
            param_dict = yaml.load(F, Loader=yaml.FullLoader)
        self.time_step = time_step
        # mass and length
        self.M = param_dict['M']
        self.G = param_dict['G']
        self.L = param_dict['L']
        # thrust and torque coefficient,
        self.CT = param_dict['CT']
        self.CM = param_dict['CM']
        thrust = self.CT
        torque_xy = self.CT * self.L / np.sqrt(2)
        torque_z = self.CM
        row_weight = np.array([[thrust], [torque_xy], [torque_xy], [torque_z]])
        matrix = np.array([[1, 1, 1, 1],
                           [-1, -1, 1, 1],
                           [-1, 1, 1, -1],
                           [-1, 1, -1, 1]])
        self.MATRIX = matrix * row_weight
        self.MATRIX_INV = np.linalg.inv(self.MATRIX)
        # Moment of inertia
        self.J = param_dict['J']
        self.J_xx = self.J[0][0]
        self.J_yy = self.J[1][1]
        self.J_zz = self.J[2][2]
        # Motor coefficient
        self.MOTOR_COEFFICIENT = param_dict['MOTOR_COEFFICIENT']
        self.MOTOR_BIAS = param_dict['MOTOR_BIAS']
        self.DRAG = param_dict['DRAG']
        # Load file
        path = self.path+'/File/cf2x.urdf'
        self.id = p.loadURDF(fileName=path,
                             basePosition=base_pos,
                             baseOrientation=base_ori,
                             physicsClientId=self.client,
                             flags=p.URDF_USE_INERTIA_FROM_FILE,
                             )
        p.changeDynamics(bodyUniqueId=self.id,
                         linkIndex=-1,
                         linearDamping=0.,
                         angularDamping=0.)
        # Motor speed and its integrator
        self.motor_speed = np.array([1., 1., 1., 1.])*(self.M*self.G/4/self.CT)**0.5

        self.integrator = ode(self.motor_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.motor_speed)

    def motor_dot(self, t, speed, speed_d):
        speed_dot = (speed_d - speed) / self.DRAG
        return speed_dot

    def apply_action(self, a, t):
        F = np.array([self.M * self.G, 0, 0, 20 * self.J_zz * a])
        motor_speed_d_pow = np.matmul(self.MATRIX_INV, F)
        h = (motor_speed_d_pow ** 0.5 - self.MOTOR_BIAS) / self.MOTOR_COEFFICIENT

        motor_speed_d = self.MOTOR_COEFFICIENT * h + self.MOTOR_BIAS
        self.integrator.set_f_params(motor_speed_d)
        self.motor_speed = self.integrator.integrate(t, t + self.time_step)
        thrust_torque = np.dot(self.MATRIX, self.motor_speed ** 2)
        torque = thrust_torque[1:]
        torque[: 2] = 0

        # apply force and torque
        p.applyExternalTorque(objectUniqueId=self.id,
                              linkIndex=-1,
                              torqueObj=torque,
                              flags=p.LINK_FRAME)
