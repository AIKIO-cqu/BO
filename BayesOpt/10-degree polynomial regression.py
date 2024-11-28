import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 生成任务的合成数据
np.random.seed(42)
sigma_epsilons = np.linspace(0.01, 0.5, 100)  # 控制器噪声级别


# 定义真实的奖励函数（ground truth）
def true_reward(sigma_epsilon):
    return 500 * (1 - np.exp(-10 * sigma_epsilon)) + np.random.normal(
        0, 10, size=sigma_epsilon.shape
    )


# 生成奖励数据
rewards = true_reward(sigma_epsilons)

# 使用多项式回归模型（10阶）拟合 \hat{g}（奖励函数的估计值）
poly_model = make_pipeline(PolynomialFeatures(degree=10), LinearRegression())
poly_model.fit(sigma_epsilons.reshape(-1, 1), rewards)
estimated_rewards = poly_model.predict(sigma_epsilons.reshape(-1, 1))

# 计算噪声水平（真实奖励与估计奖励的绝对差值）
noise_levels = np.abs(rewards - estimated_rewards)

# 使用多项式回归模型（5阶）拟合噪声水平 \sigma_\nu
poly_noise_model = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
poly_noise_model.fit(sigma_epsilons.reshape(-1, 1), noise_levels)
estimated_noise = poly_noise_model.predict(sigma_epsilons.reshape(-1, 1))

# 绘制结果
plt.figure(figsize=(14, 6))

# 绘制奖励函数的图像（左子图）
plt.subplot(1, 2, 1)
plt.scatter(
    sigma_epsilons, rewards, label="True Reward g", color="black", s=10, alpha=0.7
)  # 真实奖励数据点
plt.plot(
    sigma_epsilons,
    estimated_rewards,
    label="Estimated Reward $\hat{g}$",
    color="red",
    linewidth=2,
)  # 拟合的奖励曲线
plt.xlabel("$\\sigma_\\epsilon$")
plt.ylabel("Expected Cumulative Reward")
plt.legend()
plt.title("Expected Reward Function")
plt.grid()

# 绘制噪声函数的图像（右子图）
plt.subplot(1, 2, 2)
plt.scatter(
    sigma_epsilons, noise_levels, label="$|g - \hat{g}|$", color="gray", s=10, alpha=0.7
)  # 真实噪声数据点
plt.plot(
    sigma_epsilons,
    estimated_noise,
    label="Fitted Noise $\\sigma_\\nu$",
    color="green",
    linewidth=2,
)  # 拟合的噪声曲线
plt.xlabel("$\\sigma_\\epsilon$")
plt.ylabel("Noise")
plt.legend()
plt.title("Noise Estimation")
plt.grid()

plt.tight_layout()
plt.show()
