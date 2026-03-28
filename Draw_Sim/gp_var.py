import matplotlib.pyplot as plt
import numpy as np
import GPy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

SEED = 42
TRAIN_STEPS = 25
NOISE_STD = 0.1
PRED_STEPS = 100
HETERO_DEGREE = 10

np.random.seed(SEED)


def generate_data(steps, noise_std=NOISE_STD):
    x_train = np.linspace(0, 1, steps)
    noise = np.random.normal(0, noise_std, size=x_train.shape)
    y_train = 1 - np.exp(-10 * x_train) + noise
    return x_train, 2 - y_train


def gp_trajectory_prediction(x_train, y_train, x_test):
    kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
    gp = GPy.models.GPRegression(x_train[:, None], y_train[:, None], kernel)
    gp.optimize(messages=False)
    pred_mean, pred_var = gp.predict(x_test[:, None])
    return pred_mean, pred_var, gp


def heteroscedastic_variance(x_train, residuals, x_test, degree=HETERO_DEGREE):
    residuals_squared = residuals**2
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x_train[:, None])
    model = LinearRegression().fit(x_poly, residuals_squared)

    x_poly_test = poly.transform(x_test[:, None])
    var_pred = model.predict(x_poly_test)
    return np.maximum(var_pred, 0)


def compute_prediction_error(actual, predicted):
    error = np.abs(actual - predicted)
    return np.mean(error), np.std(error)


def plot_prediction(x_train, y_train, x_pred, y_pred, y_var, y_hetero_var):
    plt.figure(figsize=(9, 5))
    plt.plot(x_train, y_train, "kx", label="Observed point")
    plt.plot(x_pred, y_pred, "firebrick", lw=2, label="GP Predicted")
    plt.fill_between(
        x_pred,
        y_pred[:, 0] - 1.96 * np.sqrt(y_var[:, 0]),
        y_pred[:, 0] + 1.96 * np.sqrt(y_var[:, 0]),
        color="skyblue",
        alpha=0.3,
        label="Homoscedastic Variance",
    )
    plt.fill_between(
        x_pred,
        y_pred[:, 0] - 1.96 * np.sqrt(y_hetero_var),
        y_pred[:, 0] + 1.96 * np.sqrt(y_hetero_var),
        color="peachpuff",
        alpha=0.5,
        label="Heteroscedastic Variance",
    )
    plt.xlabel("Controller Parameters $\Xi$", fontsize=14)
    plt.ylabel("Error Estimation $\hat{e}$", fontsize=14)
    plt.title("Objective Function $f(\Xi)$", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    x_train, y_train = generate_data(TRAIN_STEPS, noise_std=NOISE_STD)
    x_pred = np.linspace(0, 1.0, PRED_STEPS)

    y_pred, y_var, gp = gp_trajectory_prediction(x_train, y_train, x_pred)
    train_pred = gp.predict(x_train[:, None])[0].flatten()

    y_residuals = y_train - train_pred
    y_hetero_var = heteroscedastic_variance(x_train, y_residuals, x_pred)

    error_mean, error_std = compute_prediction_error(y_train, train_pred)
    print(f"标准GP - X 轴预测误差: 平均误差 = {error_mean:.4f}, 标准差 = {error_std:.4f}")

    plot_prediction(x_train, y_train, x_pred, y_pred, y_var, y_hetero_var)
