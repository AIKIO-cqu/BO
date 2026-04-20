import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

SEED = 42
SIGMA_MIN = 0.01
SIGMA_MAX = 0.5
SIGMA_STEPS = 180
N_REPEATS = 8
MEAN_POLY_DEGREE = 8
NOISE_POLY_DEGREE = 6
NOISE_FLOOR = 0.02
NOISE_SCALE = 0.13

np.random.seed(SEED)
sigma_epsilons = np.linspace(SIGMA_MIN, SIGMA_MAX, SIGMA_STEPS)


def true_mean_reward(sigma_epsilon):
    # Latent objective: smooth trend + slight nonlinearity.
    return (
        1 - np.exp(-9 * sigma_epsilon)
        + 0.06 * np.sin(8 * np.pi * sigma_epsilon)
        - 0.02 * sigma_epsilon
    )


def true_noise_std(sigma_epsilon):
    # Heteroscedastic noise: variance changes with sigma_epsilon.
    scaled = (sigma_epsilon - SIGMA_MIN) / (SIGMA_MAX - SIGMA_MIN)
    global_trend = NOISE_FLOOR + NOISE_SCALE * scaled
    local_bump = 0.05 * np.exp(-((sigma_epsilon - 0.34) / 0.07) ** 2)
    return global_trend + local_bump


def sample_observations(sigma_epsilon, n_repeats):
    mean = true_mean_reward(sigma_epsilon)
    std = true_noise_std(sigma_epsilon)
    obs = mean[:, None] + np.random.normal(
        loc=0.0, scale=std[:, None], size=(len(sigma_epsilon), n_repeats)
    )
    return obs, mean, std


def fit_poly_curve(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    return model.predict(x.reshape(-1, 1))


def plot_reward_and_noise(
    x,
    observations,
    true_mean,
    estimated_mean,
    empirical_noise,
    true_noise,
    estimated_noise,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    ax0, ax1 = axes
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.2)

    x_rep = np.repeat(x, observations.shape[1])
    y_rep = observations.reshape(-1)
    ax0.scatter(
        x_rep,
        y_rep,
        label="Noisy Observations $y$",
        color="#334155",
        s=11,
        alpha=0.28,
        edgecolors="white",
        linewidths=0.2,
        zorder=2,
    )
    # ax0.plot(
    #     x,
    #     true_mean,
    #     label="True Mean $g(\\sigma_\\epsilon)$",
    #     color="#2563eb",
    #     linewidth=2.0,
    #     zorder=3,
    # )
    ax0.plot(
        x,
        estimated_mean,
        label="Estimated Mean $\\hat{g}$",
        color="#ef4444",
        linewidth=2.4,
        zorder=4,
    )
    ax0.set_xlabel("$\\sigma_\\epsilon$")
    ax0.set_ylabel("Expected Cumulative Reward")
    ax0.set_title("Expected Reward Function", fontsize=11, pad=8)
    ax0.legend(frameon=True, framealpha=0.9, edgecolor="#cbd5e1")

    ax1.scatter(
        x,
        empirical_noise,
        label="Empirical Noise Std",
        color="#64748b",
        s=16,
        alpha=0.55,
        edgecolors="white",
        linewidths=0.4,
        zorder=2,
    )
    # ax1.plot(
    #     x,
    #     true_noise,
    #     label="True Noise Std $\\sigma_n$",
    #     color="#2563eb",
    #     linewidth=2.0,
    #     zorder=3,
    # )
    ax1.plot(
        x,
        estimated_noise,
        label="Estimated Noise Std $\\hat{\\sigma}_n$",
        color="#16a34a",
        linewidth=2.4,
        zorder=4,
    )
    ax1.set_xlabel("$\\sigma_\\epsilon$")
    ax1.set_ylabel("Noise Std")
    ax1.set_title("Noise Estimation", fontsize=11, pad=8)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor="#cbd5e1")

    fig.suptitle(
        "Heteroscedastic Surrogate Modeling: Mean + Input-Dependent Noise",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    plt.show()


observations, true_means, true_noises = sample_observations(
    sigma_epsilons, N_REPEATS
)
sample_mean = observations.mean(axis=1)
sample_std = observations.std(axis=1, ddof=1)

estimated_rewards = fit_poly_curve(
    sigma_epsilons, sample_mean, degree=MEAN_POLY_DEGREE
)
estimated_noise = fit_poly_curve(
    sigma_epsilons, sample_std, degree=NOISE_POLY_DEGREE
)
estimated_noise = np.clip(estimated_noise, 1e-6, None)

plot_reward_and_noise(
    sigma_epsilons,
    observations,
    true_means,
    estimated_rewards,
    sample_std,
    true_noises,
    estimated_noise,
)
