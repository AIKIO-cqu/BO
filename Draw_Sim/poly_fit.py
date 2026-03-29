import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

SEED = 42
SIGMA_MIN = 0.01
SIGMA_MAX = 0.5
SIGMA_STEPS = 300
REWARD_NOISE_STD = 0.1
REWARD_POLY_DEGREE = 10
NOISE_POLY_DEGREE = 5

np.random.seed(SEED)
sigma_epsilons = np.linspace(SIGMA_MIN, SIGMA_MAX, SIGMA_STEPS)


def true_reward(sigma_epsilon):
    noise = np.random.normal(0, REWARD_NOISE_STD, size=sigma_epsilon.shape)
    return 1 - np.exp(-10 * sigma_epsilon) + noise


def fit_poly_curve(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    return model.predict(x.reshape(-1, 1))


def plot_reward_and_noise(x, rewards, estimated_rewards, noise_levels, estimated_noise):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    # fig.patch.set_facecolor("#f7f9fc")

    ax0, ax1 = axes
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.2)

    ax0.scatter(
        x,
        rewards,
        label="True Reward $g$",
        color="#334155",
        s=16,
        alpha=0.55,
        edgecolors="white",
        linewidths=0.4,
        zorder=2,
    )
    ax0.plot(
        x,
        estimated_rewards,
        label="Estimated Reward $\\hat{g}$",
        color="#ef4444",
        linewidth=2.4,
        zorder=3,
    )
    ax0.set_xlabel("$\\sigma_\\epsilon$")
    ax0.set_ylabel("Expected Cumulative Reward")
    ax0.set_title("Expected Reward Function", fontsize=11, pad=8)
    ax0.legend(frameon=True, framealpha=0.9, edgecolor="#cbd5e1")

    ax1.scatter(
        x,
        noise_levels,
        label="$|g - \\hat{g}|$",
        color="#64748b",
        s=16,
        alpha=0.55,
        edgecolors="white",
        linewidths=0.4,
        zorder=2,
    )
    ax1.plot(
        x,
        estimated_noise,
        label="Fitted Noise $\\sigma_\\nu$",
        color="#16a34a",
        linewidth=2.4,
        zorder=3,
    )
    # ax1.fill_between(x, 0, estimated_noise, color="#22c55e", alpha=0.12, zorder=1)
    ax1.set_xlabel("$\\sigma_\\epsilon$")
    ax1.set_ylabel("Noise")
    ax1.set_title("Noise Estimation", fontsize=11, pad=8)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor="#cbd5e1")

    fig.suptitle("Polynomial Fitting of Reward and Noise", fontsize=13, y=1.02)
    fig.tight_layout()
    plt.show()


rewards = true_reward(sigma_epsilons)
estimated_rewards = fit_poly_curve(
    sigma_epsilons, rewards, degree=REWARD_POLY_DEGREE
)
noise_levels = np.abs(rewards - estimated_rewards)
estimated_noise = fit_poly_curve(
    sigma_epsilons, noise_levels, degree=NOISE_POLY_DEGREE
)

plot_reward_and_noise(
    sigma_epsilons,
    rewards,
    estimated_rewards,
    noise_levels,
    estimated_noise,
)
