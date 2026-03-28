import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parent / "npy"
FNAME = "baseline"  # baseline, HBO
SHAPE = "spiral"  # circle, clover, spiral, star
VIEW_AZIM = 45.0
VIEW_ELEV = 30
Z_LIM = (0, 2)
ANIM_INTERVAL_MS = 3


def q_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_data(save_dir=SAVE_DIR, fname=FNAME, shape=SHAPE):
    dt = np.load(save_dir / f"{fname}_{shape}_dt.npy")
    x = np.load(save_dir / f"{fname}_{shape}_x.npy")
    x_ref = np.load(save_dir / f"{fname}_{shape}_x_ref.npy")
    return dt, x, x_ref


dt, x, x_ref = load_data()
print(f"Running time: {dt[-1]:.2f}s")
# ============================= 3D plot =============================
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x[:, 0], x[:, 1], x[:, 2], label="track")
ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], label="target")
ax.set_zlim(*Z_LIM)
ax.view_init(azim=VIEW_AZIM, elev=VIEW_ELEV)
plt.legend()
plt.show()

# ============================= 2D plot =============================
plt.subplot(4, 1, 1)
plt.plot(dt, x[:, 0], label="x")
plt.plot(dt, x_ref[:, 0], label="x_ref")
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(dt, x[:, 1], label="y")
plt.plot(dt, x_ref[:, 1], label="y_ref")
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(dt, x[:, 2], label="z")
plt.plot(dt, x_ref[:, 2], label="z_ref")
plt.legend()
plt.subplot(4, 1, 4)
x_yaw = np.array([q_to_yaw(q) for q in x[:, 6:10]])
plt.plot(dt, x_yaw, label="yaw")
plt.plot(dt, x_ref[:, -1], label="yaw_ref")
plt.legend()
plt.show()

# ============================= Compute Error =============================
pos_error = np.sqrt(np.sum((x[:, :3] - x_ref[:, :3]) ** 2, axis=1))
yaw_error = np.degrees(np.abs(x_yaw - x_ref[:, -1]))
print("pos_error:", np.mean(pos_error))
print("yaw_error:", np.mean(yaw_error))

# ============================= Animation =============================
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_zlim(*Z_LIM)
ax.view_init(azim=VIEW_AZIM, elev=VIEW_ELEV)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# refenrence trajectory
ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], lw=1, color="#ff7f0e", label="target")

# track trajectory
line, = ax.plot([], [], [], lw=2, color="#1f77b4", label="track")
point, = ax.plot([], [], [], "o", color="#d62728", markersize=8, label="drone")

# time text
time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

ax.legend()

start_time = [None]


def update(num):
    if start_time[0] is None:
        start_time[0] = time.time()
    current_time = time.time() - start_time[0]

    # update line and point
    line.set_data(x[: num + 1, 0], x[: num + 1, 1])
    line.set_3d_properties(x[: num + 1, 2])

    point.set_data([x[num, 0]], [x[num, 1]])
    point.set_3d_properties([x[num, 2]])

    # time_text.set_text(f'Elapsed Time: {current_time:.2f} s')
    time_text.set_text(f'Time: {dt[num]:.2f}s')
    return line, point, time_text

# create animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(dt),  # number of frames
    interval=ANIM_INTERVAL_MS,  # time interval between frames
    blit=True,
    repeat=False,
)

plt.show()

# save animation (ffmpeg required)
# ani.save('drone_animation.mp4', writer='ffmpeg', fps=30)
