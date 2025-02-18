import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
from sklearn.exceptions import ConvergenceWarning

# 忽略 ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 定义噪声模型
class NoiseModel:
    def __init__(self):
        self.beta = None
        self.scaler = StandardScaler()
        self.zeta = 1e-6  # 添加一个小的正数偏移量

    def fit(self, X, residuals):
        residuals = np.maximum(residuals, self.zeta)  # 确保 residuals 中的值都是正数
        log_residuals = np.log(residuals + self.zeta)
        X_mapped = self.scaler.fit_transform(X)
        self.beta = np.linalg.lstsq(X_mapped, log_residuals, rcond=None)[0]

    def predict(self, X):
        X_mapped = self.scaler.transform(X)
        log_noise = X_mapped @ self.beta
        return np.exp(log_noise)


# Ackley函数
def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = x**2
    sum2 = np.cos(c * x)
    term1 = -a * np.exp(-b * np.sqrt(sum1))
    term2 = -np.exp(sum2)
    return term1 + term2 + a + np.exp(1)


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


if __name__ == "__main__":
    # 定义优化边界
    bounds = np.array([-5, 5])

    # 初始化样本点
    n_initial_samples = 20
    X_sample = np.random.uniform(bounds[0], bounds[1], n_initial_samples).reshape(-1, 1)
    y_sample = []
    for i in range(n_initial_samples):
        y_sample.append(ackley_function(X_sample[i]))
    y_sample = np.array(y_sample).reshape(-1, 1)

    # 回报模型：高斯过程回归
    kernel_gp = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-5
    )
    gp = GaussianProcessRegressor(kernel=kernel_gp, n_restarts_optimizer=10)

    # 噪声模型：自定义的指数噪声模型
    noise_model = NoiseModel()

    # 训练回报模型（高斯过程）
    gp.fit(X_sample, y_sample)

    # 计算残差，训练噪声模型
    residuals = np.abs(y_sample.ravel() - gp.predict(X_sample).ravel())
    noise_model.fit(X_sample, residuals)

    # 优化循环
    n_iterations = 50
    for i in range(n_iterations):
        # Step 1: 高斯过程拟合回报模型
        gp.fit(X_sample, y_sample)

        # Step 2: 更新噪声模型
        residuals = np.abs(y_sample.ravel() - gp.predict(X_sample).ravel())
        noise_model.fit(X_sample, residuals)

        # Step 3: 使用高斯过程和噪声模型预测下一点的动态噪声
        noise_alpha = noise_model.predict(X_sample).ravel() ** 2 + 1e-6  # 动态噪声模型
        gp_with_noise = GaussianProcessRegressor(
            kernel=kernel_gp, alpha=noise_alpha, n_restarts_optimizer=10
        )
        gp_with_noise.fit(X_sample, y_sample)

        # Step 4: 使用采集函数选择下一个采样点
        X_candidates = np.random.uniform(bounds[0], bounds[1], 1000).reshape(-1, 1)
        y_min = np.min(y_sample)
        X_next = X_candidates[
            np.argmax(acquisition_function_EI(X_candidates, gp_with_noise, y_min))
        ]

        # Step 5: 评估目标函数
        y_next = ackley_function(X_next)

        print("Iteration {}: x={}, y = {}".format(i, X_next, y_next))

        # 更新样本集
        X_sample = np.vstack((X_sample, X_next))
        y_sample = np.vstack((y_sample, y_next))

    # 获取最优参数
    X_best = X_sample[np.argmin(y_sample)]
    y_best = np.min(y_sample)

    print(f"最优输入: {X_best}, 最优输出: {y_best}")

    # 画图
    x = np.linspace(bounds[0], bounds[1], 1000).reshape(-1, 1)
    y = ackley_function(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Ackley Function")
    plt.scatter(X_sample, y_sample, color="red", label="Samples")
    plt.scatter(X_best, y_best, color="green", label="Best Sample")
    gp_mean, gp_sigma = gp.predict(x, return_std=True)
    plt.plot(x, gp_mean, "--", color="b", label="GP Mean")
    plt.fill_between(
        x.ravel(),
        gp_mean - 1.96 * gp_sigma,
        gp_mean + 1.96 * gp_sigma,
        alpha=0.2,
        label="95% Confidence Interval",
    )
    plt.title("Ackley Function Optimization with Gaussian Process")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

    # 收敛过程
    plt.figure(figsize=(10, 6))
    plt.plot(y_sample, label="Sampled f(x)", marker="o")
    plt.title("Convergence of Bayesian Optimization")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
