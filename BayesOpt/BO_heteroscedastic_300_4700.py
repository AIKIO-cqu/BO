import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import time
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
    def __init__(self, degree=10):
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
    start = time.time()
    # 阶段 1：前300步========================================
    print("Optimizing for first 300 steps...")
    n_initial_samples_1 = 20
    X_sample_1 = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial_samples_1, 8))
    y_sample_1 = []
    for i in range(n_initial_samples_1):
        y_sample_1.append(
            simulate_trajectory_first_300_steps(target_trajectory, X_sample_1[i])
        )
    y_sample_1 = np.array(y_sample_1).reshape(-1, 1)
    # 目标函数模型：高斯过程回归
    kernel_gp = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-5
    )
    gp_1 = GaussianProcessRegressor(kernel=kernel_gp, n_restarts_optimizer=10)
    # 噪声模型：自定义的指数噪声模型
    noise_model_1 = ExponentialNoiseModel(degree=10)

    gp_1.fit(X_sample_1, y_sample_1)
    # 优化循环
    n_iterations_1 = 50
    for i in range(n_iterations_1):
        # Step 1: 训练噪声模型
        num_noise_updates = 5
        for j in range(num_noise_updates):
            residuals = np.abs(y_sample_1.ravel() - gp_1.predict(X_sample_1).ravel())
            noise_model_1.fit(X_sample_1, residuals)
        
        # Step 2: 预测下一点的动态噪声方差并更新gp的噪声方差
        noise_alpha = (
            noise_model_1.predict(X_sample_1).ravel() ** 2 + 1e-6
        )  # 动态噪声模型
        gp_1.set_params(alpha=noise_alpha)
        
        # Step 3: 训练高斯过程
        gp_1.fit(X_sample_1, y_sample_1)
        
        # Step 4: 选择下一个采样点
        X_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (1000, 8))
        y_min = np.min(y_sample_1)
        X_next = X_candidates[
            np.argmax(acquisition_function_EI(X_candidates, gp_1, y_min))
        ]
        y_next = simulate_trajectory_first_300_steps(target_trajectory, X_next)
        print("Iteration {}: error = {}".format(i, y_next))
        
        # Step 5: 更新样本集
        X_sample_1 = np.vstack((X_sample_1, X_next))
        y_sample_1 = np.vstack((y_sample_1, y_next))

    # 获取第一阶段的最优参数
    X_best_stage1 = X_sample_1[np.argmin(y_sample_1)]
    X_best_stage1 = [X_best_stage1[0], X_best_stage1[1], X_best_stage1[2], X_best_stage1[3],
                     X_best_stage1[4], X_best_stage1[5], X_best_stage1[6], X_best_stage1[7]]
    print(f"Best PID parameters for first 300 steps: {X_best_stage1}")

    # 阶段 2：后4700步========================================
    print("Optimizing for remaining 4700 steps...")
    n_initial_samples_2 = 20
    X_sample_2 = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial_samples_2, 8))
    y_sample_2 = []
    for i in range(n_initial_samples_2):
        y_sample_2.append(
            simulate_trajectory_last_4700_steps(
                target_trajectory, X_sample_2[i], X_best_stage1
            )
        )
    y_sample_2 = np.array(y_sample_2).reshape(-1, 1)
    # 目标函数模型：高斯过程回归
    gp_2 = GaussianProcessRegressor(kernel=kernel_gp, n_restarts_optimizer=10)
    # 噪声模型：自定义的指数噪声模型
    noise_model_2 = ExponentialNoiseModel(degree=10)

    gp_2.fit(X_sample_2, y_sample_2)
    # 优化循环
    n_iterations_2 = 50
    for i in range(n_iterations_2):
        # Step 1: 训练噪声模型
        num_noise_updates = 5
        for j in range(num_noise_updates):
            residuals = np.abs(y_sample_2.ravel() - gp_2.predict(X_sample_2).ravel())
            noise_model_2.fit(X_sample_2, residuals)
        
        # Step 2: 预测下一点的动态噪声方差并更新gp的噪声方差
        noise_alpha = (
            noise_model_2.predict(X_sample_2).ravel() ** 2 + 1e-6
        )  # 动态噪声模型
        gp_2.set_params(alpha=noise_alpha)
        
        # Step 3: 训练高斯过程
        gp_2.fit(X_sample_2, y_sample_2)
        
        # Step 4: 选择下一个采样点
        X_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (1000, 8))
        y_min = np.min(y_sample_2)
        X_next = X_candidates[
            np.argmax(acquisition_function_EI(X_candidates, gp_2, y_min))
        ]
        y_next = simulate_trajectory_last_4700_steps(
            target_trajectory, X_next, X_best_stage1
        )
        print("Iteration {}: error = {}".format(i, y_next))
        
        # Step 5: 更新样本集
        X_sample_2 = np.vstack((X_sample_2, X_next))
        y_sample_2 = np.vstack((y_sample_2, y_next))

    # 获取第二阶段的最优参数
    X_best_stage2 = X_sample_2[np.argmin(y_sample_2)]
    X_best_stage2 = [X_best_stage2[0], X_best_stage2[1], X_best_stage2[2], X_best_stage2[3],
                     X_best_stage2[4], X_best_stage2[5], X_best_stage2[6], X_best_stage2[7]]
    print(f"Best PID parameters for remaining 4700 steps: {X_best_stage2}")
    print(f"Total time: {time.time() - start:.2f}s")

    test_fixed_traj_300_4700(
        X_best_stage1, X_best_stage2, length=length_traj, shape_type=0
    )
    test_fixed_traj_300_4700(
        X_best_stage1, X_best_stage2, length=length_traj, shape_type=1
    )
    test_fixed_traj_300_4700(
        X_best_stage1, X_best_stage2, length=length_traj, shape_type=2
    )
