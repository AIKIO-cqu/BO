import numpy as np
from datetime import datetime
from EnvUAV.env_BO import YawControlEnv
from utils import generate_target_trajectory, test_fixed_traj_300_4700
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
from sklearn.exceptions import ConvergenceWarning

# 忽略 ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# 指数噪声模型类
class ExponentialNoiseModel(BaseEstimator, RegressorMixin):
    def __init__(self, degree=5):
        # 特征映射的多项式阶数
        self.degree = degree
        self.beta = None  # 回归系数
        self.z = None  # 缩放因子
        self.zeta = 1e-6  # 最小噪声偏移量

    def _feature_map(self, X):
        """生成多项式特征映射"""
        poly = PolynomialFeatures(degree=self.degree)
        return poly.fit_transform(X)

    def fit(self, X, residuals):
        """拟合噪声模型"""
        # 确保残差为正值并裁剪到合理范围
        residuals = np.maximum(np.abs(residuals), self.zeta + 1e-6)

        # 对数变换将非线性优化转为线性回归
        log_residuals = np.log(residuals - self.zeta)

        # 特征映射
        X_mapped = self._feature_map(X)

        # 检查是否存在异常值
        if np.any(np.isnan(log_residuals)) or np.any(np.isnan(X_mapped)):
            raise ValueError("NaN detected in inputs to least squares.")

        # 线性回归拟合 beta
        self.beta = np.linalg.lstsq(X_mapped, log_residuals, rcond=None)[0]

        # 计算缩放因子 z
        predicted_log = X_mapped @ self.beta
        self.z = np.exp(predicted_log).mean()

    def predict(self, X):
        """预测噪声"""
        X_mapped = self._feature_map(X)
        predicted_log = X_mapped @ self.beta
        return self.z * np.exp(predicted_log) + self.zeta


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


# 定义 UCB 采集函数
def acquisition_function_UCB(x, kappa=2.0):
    x = np.array(x).reshape(-1, 1)
    mu, sigma = gp.predict(x, return_std=True)
    return mu + kappa * sigma


# 模拟多控制器轨迹跟踪任务（前300步）
def simulate_trajectory_first_300_steps(targets, pid_params, ang_weight=0.5):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    env.new_PD_params(PD_params=pid_params)
    pos = []
    ang = []
    for i in range(300):  # 只计算前300步
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()
    pos = np.array(pos)
    ang = np.array(ang)

    # 计算误差
    pos_error = np.sqrt(np.sum((pos - targets[:300, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[:300, 3])))

    # 返回加权误差
    return np.mean(pos_error) + np.mean(ang_error) * ang_weight


# 模拟多控制器轨迹跟踪任务（后4700步）
def simulate_trajectory_last_4700_steps(
    targets, pid_params, X_best_stage1, ang_weight=0.3
):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    pos = []
    ang = []

    # 阶段1：前300步使用 X_best_stage1
    env.new_PD_params(PD_params=X_best_stage1)
    for i in range(300):
        target = targets[i, :]
        env.step(target)

    # 阶段2：后4700步使用阶段2优化参数 pid_params
    env.new_PD_params(PD_params=pid_params)
    for i in range(300, len(targets)):  # 计算后4700步
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()

    pos = np.array(pos)
    ang = np.array(ang)

    # 计算误差
    pos_error = np.sqrt(np.sum((pos - targets[300:5000, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[300:5000, 3])))

    # 返回加权误差
    return np.mean(pos_error) + np.mean(ang_error) * ang_weight


if __name__ == "__main__":
    # 定义优化边界
    bounds = np.array(
        [
            [0.0, 5.0],
            [0.0, 2.0],
            [0.0, 5.0],
            [0.0, 2.0],
            [15.0, 30.0],
            [5.0, 15.0],
            [10.0, 30.0],
            [0.0, 5.0],
        ]
    )

    # 目标轨迹
    shape_type_traj = 2  # 0: Circle, 1: Four-leaf clover, 2: Spiral
    length_traj = 5000  # 轨迹长度
    target_trajectory, name_traj = generate_target_trajectory(
        shape_type_traj, length_traj
    )

    # 分阶段优化
    # 阶段 1：前300步
    print("Optimizing for first 300 steps...")
    n_initial_samples = 20
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial_samples, 8))
    y_sample = []
    for i in range(n_initial_samples):
        y_sample.append(
            simulate_trajectory_first_300_steps(target_trajectory, X_sample[i])
        )
    y_sample = np.array(y_sample).reshape(-1, 1)
    # 回报模型：高斯过程回归
    kernel_gp = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-5
    )
    gp = GaussianProcessRegressor(kernel=kernel_gp, n_restarts_optimizer=10)
    # 噪声模型：自定义的指数噪声模型
    noise_model = ExponentialNoiseModel(degree=5)
    # 优化循环
    n_iterations = 50
    for i in range(n_iterations):
        # 高斯过程拟合
        gp.fit(X_sample, y_sample)
        # 更新噪声模型
        residuals = np.abs(y_sample.ravel() - gp.predict(X_sample).ravel())
        noise_model.fit(X_sample, residuals)
        # 使用采集函数选择下一个采样点
        X_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (1000, 8))
        y_min = np.min(y_sample)
        X_next = X_candidates[
            np.argmax(acquisition_function_EI(X_candidates, gp, y_min))
        ]
        # 评估目标函数
        y_next = simulate_trajectory_first_300_steps(target_trajectory, X_next)
        print("Iteration {}: error = {}".format(i, y_next))
        # 更新样本集
        X_sample = np.vstack((X_sample, X_next))
        y_sample = np.vstack((y_sample, y_next))
    # 获取第一阶段的最优参数
    X_best_stage1 = X_sample[np.argmin(y_sample)]
    print(f"Best PID parameters for first 300 steps: {X_best_stage1}")

    # 阶段 2：后4700步
    print("Optimizing for remaining 4700 steps...")
    n_initial_samples = 20
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial_samples, 8))
    y_sample = []
    for i in range(n_initial_samples):
        y_sample.append(
            simulate_trajectory_last_4700_steps(
                target_trajectory, X_sample[i], X_best_stage1
            )
        )
    y_sample = np.array(y_sample).reshape(-1, 1)
    # 优化循环
    for i in range(n_iterations):
        # 高斯过程拟合
        gp.fit(X_sample, y_sample)
        # 更新噪声模型
        residuals = np.abs(y_sample.ravel() - gp.predict(X_sample).ravel())
        noise_model.fit(X_sample, residuals)
        # 使用采集函数选择下一个采样点
        X_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (1000, 8))
        y_min = np.min(y_sample)
        X_next = X_candidates[
            np.argmax(acquisition_function_EI(X_candidates, gp, y_min))
        ]
        # 评估目标函数
        y_next = simulate_trajectory_last_4700_steps(
            target_trajectory, X_next, X_best_stage1
        )
        print("Iteration {}: error = {}".format(i, y_next))
        # 更新样本集
        X_sample = np.vstack((X_sample, X_next))
        y_sample = np.vstack((y_sample, y_next))
    # 获取第二阶段的最优参数
    X_best_stage2 = X_sample[np.argmin(y_sample)]
    print(f"Best PID parameters for remaining 4700 steps: {X_best_stage2}")

    test_fixed_traj_300_4700(
        X_best_stage1, X_best_stage2, length=length_traj, shape_type=0
    )
    test_fixed_traj_300_4700(
        X_best_stage1, X_best_stage2, length=length_traj, shape_type=1
    )
    test_fixed_traj_300_4700(
        X_best_stage1, X_best_stage2, length=length_traj, shape_type=2
    )
