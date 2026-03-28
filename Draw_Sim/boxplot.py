import matplotlib.pyplot as plt
import numpy as np

TRAJECTORIES = ["Ellipse", "Four-leaf", "Spiral"]
METRICS = ["Pos", "Ang"]
SAMPLES_PER_GROUP = 1000
X_POS_SCALE = 2
GROUP_OFFSET = 0.8
BOX_WIDTH = 0.6
BOX_LINEWIDTH = 1.2
GRID_STYLE = "--"
GRID_ALPHA = 0.6
TICK_LABEL_SIZE = 10
TITLE_SIZE = 14
SUPTITLE_SIZE = 16
SUPTITLE_Y = 1.05
COLORS = ["#C04F56", "skyblue"]
TITLES = ["Position Error (m)", "Angular Error ($^\circ$)"]

# ============================= 数据准备 =============================
# 两阶段优化对比数据
hbo_comparison = {
    "Ellipse": {
        "Pos": {
            "Single-stage": {"mean": 0.130, "std": 0.235},
            "Two-stage": {"mean": 0.137, "std": 0.233},
        },
        "Ang": {
            "Single-stage": {"mean": 0.744, "std": 0.389},
            "Two-stage": {"mean": 0.323, "std": 0.397},
        },
    },
    "Four-leaf": {
        "Pos": {
            "Single-stage": {"mean": 0.145, "std": 0.040},
            "Two-stage": {"mean": 0.162, "std": 0.052},
        },
        "Ang": {
            "Single-stage": {"mean": 3.342, "std": 1.618},
            "Two-stage": {"mean": 1.328, "std": 1.260},
        },
    },
    "Spiral": {
        "Pos": {
            "Single-stage": {"mean": 0.312, "std": 0.263},
            "Two-stage": {"mean": 0.305, "std": 0.226},
        },
        "Ang": {
            "Single-stage": {"mean": 1.369, "std": 2.562},
            "Two-stage": {"mean": 0.649, "std": 2.612},
        },
    },
}

# 噪声模型对比数据
noise_comparison = {
    "Ellipse": {
        "Pos": {
            "Polynomial": {"mean": 0.174, "std": 0.230},
            "Exponential": {"mean": 0.137, "std": 0.233},
        },
        "Ang": {
            "Polynomial": {"mean": 0.594, "std": 0.362},
            "Exponential": {"mean": 0.323, "std": 0.397},
        },
    },
    "Four-leaf": {
        "Pos": {
            "Polynomial": {"mean": 0.204, "std": 0.095},
            "Exponential": {"mean": 0.162, "std": 0.052},
        },
        "Ang": {
            "Polynomial": {"mean": 2.613, "std": 1.371},
            "Exponential": {"mean": 1.328, "std": 1.260},
        },
    },
    "Spiral": {
        "Pos": {
            "Polynomial": {"mean": 0.415, "std": 0.250},
            "Exponential": {"mean": 0.305, "std": 0.226},
        },
        "Ang": {
            "Polynomial": {"mean": 1.069, "std": 2.525},
            "Exponential": {"mean": 0.649, "std": 2.612},
        },
    },
}

# 高斯过程拟合和多项式拟合对比
fitting_comparison = {
    "Ellipse": {
        "Pos": {
            "Polynomial": {"mean": 0.147, "std": 0.233},
            "Gaussian processe": {"mean": 0.137, "std": 0.233},
        },
        "Ang": {
            "Polynomial": {"mean": 1.070, "std": 0.519},
            "Gaussian processe": {"mean": 0.323, "std": 0.397},
        },
    },
    "Four-leaf": {
        "Pos": {
            "Polynomial": {"mean": 0.167, "std": 0.050},
            "Gaussian processe": {"mean": 0.162, "std": 0.052},
        },
        "Ang": {
            "Polynomial": {"mean": 4.824, "std": 2.344},
            "Gaussian processe": {"mean": 1.328, "std": 1.260},
        },
    },
    "Spiral": {
        "Pos": {
            "Polynomial": {"mean": 0.301, "std": 0.222},
            "Gaussian processe": {"mean": 0.305, "std": 0.226},
        },
        "Ang": {
            "Polynomial": {"mean": 1.813, "std": 2.558},
            "Gaussian processe": {"mean": 0.649, "std": 2.612},
        },
    },
}


def generate_samples(mean, std, n_samples=SAMPLES_PER_GROUP):
    """生成正态分布样本"""
    return np.random.normal(mean, std, n_samples)


def _collect_metric_group_data(data_dict, metric, group_name):
    return [
        generate_samples(
            data_dict[traj][metric][group_name]["mean"],
            data_dict[traj][metric][group_name]["std"],
        )
        for traj in TRAJECTORIES
    ]


def _style_box(box, color):
    for patch in box["boxes"]:
        patch.set_facecolor(color)
    for element in ["whiskers", "caps", "medians"]:
        plt.setp(box[element], color="black", linewidth=BOX_LINEWIDTH)


def plot_dual_comparison(data_dict, titles, fig_title, labels, colors):
    """绘制双对比图（一行两列）"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(fig_title, fontsize=SUPTITLE_SIZE, y=SUPTITLE_Y)

    x_pos = np.arange(len(TRAJECTORIES)) * X_POS_SCALE

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        for group_idx, (group_name, color) in enumerate(zip(labels, colors)):
            all_data = _collect_metric_group_data(data_dict, metric, group_name)
            positions = x_pos + group_idx * GROUP_OFFSET
            box = ax.boxplot(
                all_data,
                positions=positions,
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
            )
            _style_box(box, color)

        ax.set_xticks(x_pos + GROUP_OFFSET / 2)
        ax.set_xticklabels(TRAJECTORIES)
        ax.set_title(titles[idx], fontsize=TITLE_SIZE, pad=10)
        ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

        if idx == 1:
            handles = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
            ax.legend(handles, labels, fontsize=10)

    plt.tight_layout()
    return fig


plot_dual_comparison(
    hbo_comparison,
    titles=TITLES,
    fig_title="Optimization Stage Performance Comparison",
    labels=["Single-stage", "Two-stage"],
    colors=COLORS,
)

plot_dual_comparison(
    noise_comparison,
    titles=TITLES,
    fig_title="Noise Model Performance Comparison",
    labels=["Polynomial", "Exponential"],
    colors=COLORS,
)

plot_dual_comparison(
    fitting_comparison,
    titles=TITLES,
    fig_title="Fitting Performance Comparison",
    labels=["Polynomial", "Gaussian processe"],
    colors=COLORS,
)

plt.show()
