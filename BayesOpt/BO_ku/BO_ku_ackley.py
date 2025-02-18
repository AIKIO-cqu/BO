import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization


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


# 包装目标函数用于Bayesian Optimization
def target_function(x):
    return -ackley_function(x)  # 最大化需要对目标函数取负


if __name__ == "__main__":
    # 定义优化边界
    pounds = {"x": (-5, 5)}

    optimizer = BayesianOptimization(
        f=target_function,
        pbounds=pounds,
        verbose=2,  # 打印优化日志
        random_state=42,
    )

    # 执行优化
    optimizer.maximize(
        init_points=20,  # 初始随机采样点的数量
        n_iter=50,  # 优化迭代次数
    )

    # 提取最优结果
    x_best = optimizer.max["params"]["x"]
    y_best = -optimizer.max["target"]  # 还原为最小化目标
    print(f"最优输入: x = {x_best}, 最优输出: y = {y_best}")

    # 画出优化过程中的样本点
    x = np.linspace(-5, 5, 1000)
    y = ackley_function(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Ackley Function")
    samples_x = [res["params"]["x"] for res in optimizer.res]
    samples_y = [-res["target"] for res in optimizer.res]
    plt.scatter(samples_x, samples_y, color="red", label="Samples")
    plt.scatter(x_best, y_best, color="green", label="Best Sample")
    gp_mean, gp_sigma = optimizer._gp.predict(x.reshape(-1, 1), return_std=True)
    plt.plot(x, -gp_mean, "--", color="b", label="GP Mean")
    plt.fill_between(
        x,
        -gp_mean - 1.96 * gp_sigma,
        -gp_mean + 1.96 * gp_sigma,
        alpha=0.2,
        label="95% Confidence Interval",
    )
    plt.title("Ackley Function Optimization with Bayesian Optimization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

    # 收敛过程
    plt.figure(figsize=(10, 6))
    plt.plot(samples_y, label="Sampled f(x)", marker="o")
    plt.title("Convergence of Bayesian Optimization")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
