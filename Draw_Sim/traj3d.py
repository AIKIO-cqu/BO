import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "npy"
TRAJ_LENGTH = 5000
VIEW_AZIM = 45.0
VIEW_ELEV = 30

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
        # Some datasets use tiny text alias files that point to another .npy file.
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


def _load_controller_trajs(shape_suffix):
    return {
        label: load_npy(f"pos_{prefix}_{shape_suffix}.npy")
        for prefix, label in CONTROLLER_LABELS
    }


def _draw_shape(shape_type, shape_suffix):
    targets, _ = generate_target_trajectory(shape_type, TRAJ_LENGTH)
    controller_trajs = _load_controller_trajs(shape_suffix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for _, label in CONTROLLER_LABELS:
        traj = controller_trajs[label]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=label)

    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], label="Ref Traj", color="black")
    ax.view_init(azim=VIEW_AZIM, elev=VIEW_ELEV)
    # plt.legend()
    plt.show()


def draw_Ellipse():
    _draw_shape(shape_type=0, shape_suffix="Ellipse")


def draw_Four():
    _draw_shape(shape_type=1, shape_suffix="Four")


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
