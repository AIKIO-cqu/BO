import numpy as np
import matplotlib.pyplot as plt
import torch
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace


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
    # 定义HEBO的搜索空间
    space = DesignSpace().parse(
        [
            {"name": "x", "type": "num", "lb": -5.0, "ub": 5.0},  # 定义x的范围
        ]
    )

    # 初始化HEBO优化器
    hebo_optimizer = HEBO(space)

    init_points = 20  # 初始随机采样点数量
    n_iter = 50  # 优化迭代次数

    # 初始采样点
    initial_samples = hebo_optimizer.suggest(n_suggestions=init_points)
    y_init = np.array(
        [target_function(sample["x"]) for _, sample in initial_samples.iterrows()]
    )
    hebo_optimizer.observe_new_data(initial_samples, y_init)

    # 开始迭代优化
    for i in range(n_iter):
        # 生成新的候选采样点
        suggestions = hebo_optimizer.suggest(n_suggestions=1)
        # 计算目标函数值
        y_suggestions = np.array(
            [target_function(suggestion["x"]) for _, suggestion in suggestions.iterrows()]
        )
        # 更新HEBO模型
        hebo_optimizer.observe_new_data(suggestions, y_suggestions)

        print(f"Iteration {i + 1}/{n_iter}, Best y: {-hebo_optimizer.best_y}")
    
    # 提取最优结果
    x_best = hebo_optimizer.best_x.iloc[0]["x"]
    y_best = -hebo_optimizer.best_y
    print(f"最优输入: x = {x_best}, 最优输出: y = {y_best}")

    # 画出优化过程中的样本点
    x = np.linspace(-5, 5, 1000)
    y = ackley_function(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Ackley Function")
    samples_x = hebo_optimizer.X["x"].values
    samples_y = -hebo_optimizer.y.ravel()
    plt.scatter(samples_x, samples_y, color="red", label="Samples")
    plt.scatter(x_best, y_best, color="green", label="Best Sample")
    gp_mean, gp_sigma = hebo_optimizer.model.predict(torch.tensor(x).reshape(-1, 1))
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
