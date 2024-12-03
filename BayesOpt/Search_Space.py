import numpy as np
from EnvUAV.env_BO import YawControlEnv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.stats import norm
from test_params import test_params
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# 定义 EI 采集函数
def acquisition_function_EI(x, gp, y_min):
    mu, sigma = gp.predict(x, return_std=True)
    sigma = sigma.ravel()  # 转为 1 维
    improvement = y_min - mu.ravel()  # 计算改进
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.zeros_like(improvement)
        valid = sigma > 0  # 避免除零
        Z[valid] = improvement[valid] / sigma[valid]
        ei = np.zeros_like(improvement)
        ei[valid] = improvement[valid] * norm.cdf(Z[valid]) + sigma[valid] * norm.pdf(
            Z[valid]
        )
    return ei


class Optimizer:
    def __init__(self, env):
        self.env = env

    def set_attitude_controller(self, P, D):
        self.env.attitude_controller.set_param(P, D)

    def set_z_controller(self, P, D):
        self.env.z_controller.set_param(P, D)

    def set_xy_controller(self, P, D):
        self.env.x_controller.set_param(P, D)
        self.env.y_controller.set_param(P, D)

    def printPID(self):
        print("x_controller: ", self.env.x_controller.P, self.env.x_controller.D)
        print("y_controller: ", self.env.y_controller.P, self.env.y_controller.D)
        print("z_controller: ", self.env.z_controller.P, self.env.z_controller.D)
        print(
            "attitude_controller: ",
            self.env.attitude_controller.P,
            self.env.attitude_controller.D,
        )

    def objective(self, params, tag):
        env = self.env

        if tag == "attitude":
            env.attitude_controller.set_param(params[0], params[1])
        elif tag == "z":
            env.z_controller.set_param(params[0], params[1])
        elif tag == "xy":
            env.x_controller.set_param(params[0], params[1])
            env.y_controller.set_param(params[0], params[1])

        pos = []
        ang = []
        env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, np.pi / 3]))
        targets = np.array([[0, 0, 0, 0]])
        for episode in range(len(targets)):
            target = targets[episode, :]
            for ep_step in range(500):
                env.step(target)
                pos.append(env.current_pos.tolist())
                ang.append(env.current_ori.tolist())
        env.close()
        pos = np.array(pos)
        ang = np.array(ang)
        # 位置误差、角度误差
        pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
        ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
        return np.mean(pos_error) + np.mean(ang_error) * 0.1
        # return np.mean(pos_error)

    def optimize(self, bounds, tag):
        # 初始化样本点
        n_initial_samples = 10
        X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial_samples, 2))
        y_sample = []
        for i in range(n_initial_samples):
            y_sample.append(self.objective(X_sample[i], tag))
        y_sample = np.array(y_sample).reshape(-1, 1)

        # 定义高斯过程回归器
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
            noise_level=1e-5
        )
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # 优化循环
        n_iterations = 100
        for i in range(n_iterations):
            # 高斯过程拟合
            gp.fit(X_sample, y_sample)

            # 选择下一个采样点
            X_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (1000, 2))
            y_min = np.min(y_sample)
            X_next = X_candidates[
                np.argmax(acquisition_function_EI(X_candidates, gp, y_min))
            ]

            # 评估目标函数
            y_next = self.objective(X_next, tag)

            print("Iteration {}: error = {}".format(i, y_next))

            # 更新样本集
            X_sample = np.vstack((X_sample, X_next))
            y_sample = np.vstack((y_sample, y_next))

        # 获取最优参数
        X_best = X_sample[np.argmin(y_sample)]
        y_best = np.min(y_sample)

        print(f"最优输入: {X_best}, 最优输出: {y_best}")

        return X_best


if __name__ == "__main__":
    bounds_attitude = np.array([[10, 30], [0, 5]])
    bounds_z = np.array([[10, 30], [5, 15]])
    bounds_xy = np.array([[0, 5], [0, 5]])

    optimizer = Optimizer(YawControlEnv())
    optimizer.printPID()

    attitude_params = optimizer.optimize(bounds_attitude, "attitude")
    optimizer.set_attitude_controller(attitude_params[0], attitude_params[1])
    optimizer.printPID()

    z_params = optimizer.optimize(bounds_z, "z")
    optimizer.set_z_controller(z_params[0], z_params[1])
    optimizer.printPID()

    xy_params = optimizer.optimize(bounds_xy, "xy")
    optimizer.set_xy_controller(xy_params[0], xy_params[1])
    optimizer.printPID()

    optimized_params = np.array(
        [
            xy_params[0],
            xy_params[1],
            xy_params[0],
            xy_params[1],
            z_params[0],
            z_params[1],
            attitude_params[0],
            attitude_params[1],
        ]
    )
    test_params(optimized_params)
