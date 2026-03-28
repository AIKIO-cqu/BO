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
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(x, rewards, label="True Reward g", color="black", s=10, alpha=0.7)
    plt.plot(
        x,
        estimated_rewards,
        label="Estimated Reward $\hat{g}$",
        color="red",
        linewidth=2,
    )
    plt.xlabel("$\\sigma_\\epsilon$")
    plt.ylabel("Expected Cumulative Reward")
    plt.legend()
    plt.title("Expected Reward Function")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(
        x,
        noise_levels,
        label="$|g - \hat{g}|$",
        color="gray",
        s=10,
        alpha=0.7,
    )
    plt.plot(
        x,
        estimated_noise,
        label="Fitted Noise $\\sigma_\\nu$",
        color="green",
        linewidth=2,
    )
    plt.xlabel("$\\sigma_\\epsilon$")
    plt.ylabel("Noise")
    plt.legend()
    plt.title("Noise Estimation")
    plt.grid()

    plt.tight_layout()
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
