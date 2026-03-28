import torch
import torch.nn as nn
import numpy as np
import pybullet as p
import casadi as ca


def shift(u, x_n):
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return u_end, x_n


class HeightMPC:
    def __init__(self, T=0.01, N=40, Q=np.diag([14, 1]), R=np.diag([1])):
        self.T = T  # time step
        self.N = N  # horizon length

        # weight matrix
        self.Q = Q
        self.R = R

        # The history states and controls
        self.next_states = np.zeros((self.N + 1, 2))
        self.u0 = np.zeros((self.N, 1))

        self.setupController()

    def setupController(self):
        self.opti = ca.Opti()
        # state and input
        self.opt_controls = self.opti.variable(self.N, 1)
        self.opt_states = self.opti.variable(self.N + 1, 2)

        # create model
        f = lambda x_, u_: ca.vertcat(*[
            x_[1],
            5 * u_])

        # parameters, these parameters are the reference and initial state
        self.opt_x_ref = self.opti.parameter(1, 2)
        self.opt_x_init = self.opti.parameter(1, 2)

        # initial condition
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_init)
        # dynamic constrain
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T * self.T
            self.opti.subject_to(self.opt_states[i + 1, :] == x_next)

        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i, :] - self.opt_x_ref
            control_error_ = self.opt_controls[i, :]
            obj = obj + ca.mtimes([state_error_, self.Q, state_error_.T]) \
                  + ca.mtimes([control_error_, self.R, control_error_.T])
        self.opti.minimize(obj)

        self.opti.subject_to(self.opti.bounded(-1, self.opt_controls, 1))

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_init, x_ref):
        # set parameter, here only update initial state of x (x0)
        self.opti.set_value(self.opt_x_ref, x_ref)
        self.opti.set_value(self.opt_x_init, x_init)

        # provide the initial guess of the optimization targets
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0.reshape(self.N, 1))

        # solve the problem
        sol = self.opti.solve()

        # obtain the control input
        u_res = sol.value(self.opt_controls)
        x_m = sol.value(self.opt_states)
        self.u0, self.next_states = shift(u_res, x_m)
        return u_res[0]


class AttitudeMPC:
    def __init__(self, T=0.01, N=20, Q=np.diag([100, 1]), R=np.diag([1])):
        self.T = T  # time step
        self.N = N  # horizon length

        # weight matrix
        self.Q = Q
        self.R = R

        # The history states and controls
        self.next_states = np.zeros((self.N + 1, 2))
        self.u0 = np.zeros((self.N, 1))

        self.setupController()

    def setupController(self):
        self.opti = ca.Opti()
        # state and input
        self.opt_controls = self.opti.variable(self.N, 1)
        self.opt_states = self.opti.variable(self.N + 1, 2)

        # create model
        f = lambda x_, u_: ca.vertcat(*[
            x_[1],
            20 * u_])

        # parameters, these parameters are the reference and initial state
        self.opt_x_ref = self.opti.parameter(1, 2)
        self.opt_x_init = self.opti.parameter(1, 2)

        # initial condition
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_init)
        # dynamic constrain
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T * self.T
            self.opti.subject_to(self.opt_states[i + 1, :] == x_next)

        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i, :] - self.opt_x_ref
            control_error_ = self.opt_controls[i, :]
            obj = obj + ca.mtimes([state_error_, self.Q, state_error_.T]) \
                  + ca.mtimes([control_error_, self.R, control_error_.T])
        self.opti.minimize(obj)

        self.opti.subject_to(self.opti.bounded(-1, self.opt_controls, 1))

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_init, x_ref):
        # set parameter, here only update initial state of x (x0)
        self.opti.set_value(self.opt_x_ref, x_ref)
        self.opti.set_value(self.opt_x_init, x_init)

        # provide the initial guess of the optimization targets
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0.reshape(self.N, 1))

        # solve the problem
        sol = self.opti.solve()

        # obtain the control input
        u_res = sol.value(self.opt_controls)
        x_m = sol.value(self.opt_states)
        self.u0, self.next_states = shift(u_res, x_m)
        return u_res[0]


class PositionMPC:
    def __init__(self, T=0.01, N=40, Q=np.diag([7.7, 1]), R=np.diag([1])):
        self.T = T  # time step
        self.N = N  # horizon length

        # weight matrix
        self.Q = Q
        self.R = R

        # The history states and controls
        self.next_states = np.zeros((self.N + 1, 2))
        self.u0 = np.zeros((self.N, 1))

        self.setupController()

    def setupController(self):
        self.opti = ca.Opti()
        # state and input
        self.opt_controls = self.opti.variable(self.N, 1)
        self.opt_states = self.opti.variable(self.N + 1, 2)

        # create model
        f = lambda x_, u_: ca.vertcat(*[
            x_[1],
            5 * u_])

        # parameters, these parameters are the reference and initial state
        self.opt_x_ref = self.opti.parameter(1, 2)
        self.opt_x_init = self.opti.parameter(1, 2)

        # initial condition
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_init)
        # dynamic constrain
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T * self.T
            self.opti.subject_to(self.opt_states[i + 1, :] == x_next)

        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i, :] - self.opt_x_ref
            control_error_ = self.opt_controls[i, :]
            obj = obj + ca.mtimes([state_error_, self.Q, state_error_.T]) \
                  + ca.mtimes([control_error_, self.R, control_error_.T])
        self.opti.minimize(obj)

        self.opti.subject_to(self.opti.bounded(-1, self.opt_controls, 1))

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_init, x_ref):
        # set parameter, here only update initial state of x (x0)
        self.opti.set_value(self.opt_x_ref, x_ref)
        self.opti.set_value(self.opt_x_init, x_init)

        # provide the initial guess of the optimization targets
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0.reshape(self.N, 1))

        # solve the problem
        sol = self.opti.solve()

        # obtain the control input
        u_res = sol.value(self.opt_controls)
        x_m = sol.value(self.opt_states)
        self.u0, self.next_states = shift(u_res, x_m)
        return u_res[0]