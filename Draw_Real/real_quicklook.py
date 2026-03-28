from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = Path(__file__).resolve().parent / "npy"
FNAME = "HBO_circle"


def load_data(fname=FNAME):
    dt = np.load(SAVE_DIR / f"{fname}_dt.npy")
    x = np.load(SAVE_DIR / f"{fname}_x.npy")
    x_ref = np.load(SAVE_DIR / f"{fname}_x_ref.npy")
    return dt, x, x_ref


def compute_rmse(x, x_ref):
    diff = x[:, :3] - x_ref[:, :3]
    return np.sum(np.sqrt(np.mean(diff**2)))


def plot_quicklook(dt, x, x_ref):
    plt.plot(dt, x[:, 0], label="x")
    plt.plot(dt, x_ref[:, 0], label="x_ref")
    plt.legend()

    plt.figure()
    plt.gca().set_aspect("equal")
    plt.plot(x[:, 0], x[:, 1], label="real")
    plt.plot(x_ref[:, 0], x_ref[:, 1], label="des")
    plt.legend()
    plt.show()


def main():
    dt, x, x_ref = load_data()
    print("rmse:", compute_rmse(x, x_ref))
    plot_quicklook(dt, x, x_ref)


if __name__ == "__main__":
    main()
