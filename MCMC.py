import numpy as np
import scipy.stats
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def MCMC(P, X0, chain, space):
    """
    P: 随机变量X服从的概率密度函数
    X0: MCMC链的初始值
    chain: MCMC链的长度
    space: 随机变量的取值范围，例如[0, inf]
    """

    if not callable(P):
        raise ValueError("P must be a function")

    X_current = X0
    X = [X_current]

    while True:
        Delta_X = scipy.stats.norm(loc=0, scale=2).rvs()
        X_proposed = X_current + Delta_X

        if X_proposed < space[0] or X_proposed > space[1]:
            p_moving = 0
        elif P(X_current) == 0:
            p_moving = 1
        else:
            p_moving = min(1, P(X_proposed) / P(X_current))

        if scipy.stats.uniform.rvs() <= p_moving:
            X.append(X_proposed)
            X_current = X_proposed
        else:
            X.append(X_current)

        if len(X) >= chain:
            break
    return np.array(X)


def GammaDist(x, k, theta):
    return 1 / (gamma(k) * theta**k) * x ** (k - 1) * np.exp(-x / theta)


def main():
    gamma_dist = lambda x: GammaDist(x, 2, 2)
    X = MCMC(gamma_dist, 2, 1000, [0, np.inf])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("MCMC Process")

    def anima_chain(index):
        ax1.clear()
        ax1.plot(X[: index + 1], "r-")
        ax1.set_xlabel("Chain Length")
        ax1.set_ylabel("X Value")

    def anima_density(index):
        ax2.clear()
        x = np.arange(0, 12, 0.1)
        ax2.plot(x, gamma_dist(x), "k-")
        if index <= 10:
            y = np.zeros(len(x))
            ax2.plot(x, y)
        else:
            density = scipy.stats.gaussian_kde(X[: index + 1])
            ax2.plot(x, density(x), "r-")
        ax2.set_xlim([0, 12])
        ax2.set_ylim([0, 0.5])
        ax2.set_xlabel("X Value")
        ax2.set_ylabel("Density")

    ani_chain = animation.FuncAnimation(fig, anima_chain, frames=1000, interval=20)
    ani_density = animation.FuncAnimation(fig, anima_density, frames=1000, interval=20)
    plt.show()


if __name__ == "__main__":
    main()
