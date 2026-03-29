import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "npy_exchange"
TRAJ_LENGTH = 5000

CONTROLLER_LABELS = [
    ("PID", "PID"),
    ("LQR", "LQR"),
    ("MPC", "MPC"),
    ("TD3", "TD3"),
    ("PID+RS", "RS-PID"),
    ("PID+BO", "BO-PID"),
    ("PID+HBO", "HBO-PID"),
]


def load_npy(filename, _seen=None):
    path = DATA_DIR / filename
    try:
        return np.load(path)
    except ValueError as exc:
        # Some datasets store a tiny text alias that points to another .npy file.
        # Resolve those aliases to keep loading robust across environments.
        if _seen is None:
            _seen = set()
        if path in _seen:
            raise ValueError(f"Circular npy alias detected: {path}") from exc
        _seen.add(path)

        data = path.read_bytes()
        if len(data) > 256:
            raise

        target = data.decode("utf-8", errors="ignore").strip()
        if not target.endswith(".npy"):
            raise
        return load_npy(target, _seen=_seen)


# 生成目标轨迹
def generate_target_trajectory(shape_type, length):
    if shape_type == 0:
        name = "Ellipse"  # 圆形
        index = np.array(range(length)) / length
        tx = 2 * np.cos(2 * np.pi * index)
        ty = 2 * np.sin(2 * np.pi * index)
        tz = -np.cos(2 * np.pi * index) - np.sin(2 * np.pi * index)
        tpsi = np.sin(2 * np.pi * index) * np.pi / 3 * 2
    elif shape_type == 1:
        name = "Four-leaf clover"  # 四叶草形状
        index = np.array(range(length)) / length * 2
        tx = 2 * np.sin(2 * np.pi * index) * np.cos(np.pi * index)
        ty = 2 * np.sin(2 * np.pi * index) * np.sin(np.pi * index)
        tz = -np.sin(2 * np.pi * index) * np.cos(np.pi * index) - np.sin(
            2 * np.pi * index
        ) * np.sin(np.pi * index)
        tpsi = np.sin(4 * np.pi * index) * np.pi / 4 * 3
    elif shape_type == 2:
        name = "Spiral"  # 半径先增大后减小的螺旋形状
        index = np.array(range(length)) / length * 4  # 轨迹参数，4 圈
        radius = 2 + 0.3 * np.sin(np.pi * index)  # 半径先增大后减小
        tx = radius * np.cos(1.5 * np.pi * index)  # x 方向的螺旋
        ty = radius * np.sin(1.5 * np.pi * index)  # y 方向的螺旋
        tz = 0.5 * index - 1  # z 方向逐渐上升
        tpsi = np.cos(2 * np.pi * index) * np.pi / 4  # 偏航角周期变化
    else:
        raise ValueError("shape_type must be 0, 1, or 2")
    target_trajectory = np.vstack([tx, ty, tz, tpsi]).T

    return target_trajectory, name


def _load_controller_series(shape_suffix):
    return {
        label: {
            "pos": load_npy(f"pos_{prefix}_{shape_suffix}.npy"),
            "ang": load_npy(f"ang_{prefix}_{shape_suffix}.npy"),
        }
        for prefix, label in CONTROLLER_LABELS
    }


def _draw_shape(shape_type, shape_suffix, title_suffix=None):
    targets, _ = generate_target_trajectory(shape_type, TRAJ_LENGTH)
    controller_series = _load_controller_series(shape_suffix)

    n_steps = targets.shape[0]
    for _, label in CONTROLLER_LABELS:
        n_steps = min(
            n_steps,
            controller_series[label]["pos"].shape[0],
            controller_series[label]["ang"].shape[0],
        )
    dt = np.arange(n_steps)
    targets = targets[:n_steps]

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    color_map = {
        label: default_colors[idx % len(default_colors)]
        for idx, (_, label) in enumerate(CONTROLLER_LABELS)
    }

    fig = plt.figure(figsize=(10, 10))
    title = title_suffix if title_suffix is not None else shape_suffix
    fig.suptitle(f"{title} Comparison")

    ax1 = plt.subplot(4, 1, 1)
    ax2 = plt.subplot(4, 1, 2)
    ax3 = plt.subplot(4, 1, 3)
    ax4 = plt.subplot(4, 1, 4)

    ax1.plot(dt, targets[:, 0], label="Ref Traj", color="black")
    ax2.plot(dt, targets[:, 1], label="Ref Traj", color="black")
    ax3.plot(dt, targets[:, 2], label="Ref Traj", color="black")
    ax4.plot(dt, targets[:, -1], label="Ref Traj", color="black")

    for _, label in CONTROLLER_LABELS:
        pos = controller_series[label]["pos"][:n_steps]
        yaw = controller_series[label]["ang"][:n_steps, 2]
        color = color_map[label]
        ax1.plot(dt, pos[:, 0], label=f"{label}", color=color)
        ax2.plot(dt, pos[:, 1], label=f"{label}", color=color)
        ax3.plot(dt, pos[:, 2], label=f"{label}", color=color)
        ax4.plot(dt, yaw, label=f"{label}", color=color)

    # ax1.legend(loc="upper right")
    # ax2.legend(loc="upper right")
    # ax3.legend(loc="upper right")
    # ax4.legend(loc="upper right")

    ax4.set_xlabel("Time (step)")
    ax1.set_ylabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax3.set_ylabel("z (m)")
    ax4.set_ylabel("yaw (rad)")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def draw_Ellipse():
    _draw_shape(shape_type=0, shape_suffix="Ellipse")


def draw_Four():
    _draw_shape(shape_type=1, shape_suffix="Four", title_suffix="Four-leaf clover")


def draw_Spiral():
    _draw_shape(shape_type=2, shape_suffix="Spiral")


if __name__ == "__main__":
    draw_Ellipse()
    draw_Four()
    draw_Spiral()


# 测试用的参数
PID_params = [1, 0.77, 1, 0.77, 20, 10.5, 20, 3.324]
PID_RS_params = [
    2.074697139425429,
    1.8002557547010822,
    3.2318102503303674,
    1.6096471309988258,
    22.124103912645428,
    12.188682900042139,
    17.760882764135705,
    2.559802781742307,
]
PID_BO_params = [
    1.4903032285039675,
    0.6555869336657205,
    3.7072932982417224,
    1.8422600068867023,
    20.62216163513675,
    13.64999389298014,
    19.4343066034699,
    2.62139093544266,
]
PID_HBO_params = [
    2.4072319861877944,
    1.1177837202417729,
    0.931257950917781,
    0.4007567734145976,
    15.18091718709083,
    5.013179923390205,
    27.713004830064556,
    3.1048918772723426,
]
