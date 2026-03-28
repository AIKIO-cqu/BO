import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.ticker as ticker
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parent / "npy"
FNAME_BASELINE = "baseline"
FNAME_HBO = "HBO"
SHAPE = "circle"  # circle, clover, spiral, star
N_STEPS = 2000
VIEW_AZIM = 45.0
VIEW_ELEV = 30
Z_LIM = (0, 2)
ANIM_INTERVAL_MS = 3


def q_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_npy_series(prefix, shape, n_steps=N_STEPS):
    dt = np.load(SAVE_DIR / f"{prefix}_{shape}_dt.npy")[:n_steps]
    x = np.load(SAVE_DIR / f"{prefix}_{shape}_x.npy")[:n_steps]
    x_ref = np.load(SAVE_DIR / f"{prefix}_{shape}_x_ref.npy")[:n_steps]
    return dt, x, x_ref


def set_3d_axes(ax):
    ax.set_zlim(*Z_LIM)
    ax.zaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.view_init(azim=VIEW_AZIM, elev=VIEW_ELEV)


dt, x_baseline, x_ref = load_npy_series(FNAME_BASELINE, SHAPE)
x_HBO = np.load(SAVE_DIR / f"{FNAME_HBO}_{SHAPE}_x.npy")[:N_STEPS]
print(f"Running time: {dt[-1]:.2f}s")

# ============================= Compute Error =============================
x_yaw_baseline = np.array([q_to_yaw(x_baseline[i, 6:10]) for i in range(len(dt))])
x_yaw_HBO = np.array([q_to_yaw(x_HBO[i, 6:10]) for i in range(len(dt))])
pos_error_baseline = np.sqrt(np.sum((x_baseline[:, :3] - x_ref[:, :3]) ** 2, axis=1))
yaw_error_baseline = np.degrees(np.abs(x_yaw_baseline - x_ref[:, -1]))
pos_error_HBO = np.sqrt(np.sum((x_HBO[:, :3] - x_ref[:, :3]) ** 2, axis=1))
yaw_error_HBO = np.degrees(np.abs(x_yaw_HBO - x_ref[:, -1]))
print(f"pos_error_baseline: {np.mean(pos_error_baseline):.3f}")
print(f"yaw_error_baseline: {np.mean(yaw_error_baseline):.3f}")
print(f"pos_error_HBO: {np.mean(pos_error_HBO):.3f}")
print(f"yaw_error_HBO: {np.mean(yaw_error_HBO):.3f}")

# ============================= 3D plot =============================
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], label="ref")
ax.plot(x_baseline[:, 0], x_baseline[:, 1], x_baseline[:, 2], label="BO")
ax.plot(x_HBO[:, 0], x_HBO[:, 1], x_HBO[:, 2], label="HBO")
set_3d_axes(ax)
# plt.legend()
plt.show()

# ============================= Animation =============================
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
set_3d_axes(ax)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# refenrence trajectory
ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], lw=1, color="#1f77b4", label="ref")

# track trajectory
line_baseline, = ax.plot([], [], [], lw=2, color="#ff7f0e", label="baseline")
point_baseline, = ax.plot([], [], [], "o", color="#ff7f0e", markersize=8)
line_HBO, = ax.plot([], [], [], lw=2, color="#2ca02c", label="HBO")
point_HBO, = ax.plot([], [], [], "o", color="#2ca02c", markersize=8)

# time text
time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

ax.legend()

def update(num):
    # update line and point
    line_baseline.set_data(x_baseline[: num + 1, 0], x_baseline[: num + 1, 1])
    line_baseline.set_3d_properties(x_baseline[: num + 1, 2])

    point_baseline.set_data([x_baseline[num, 0]], [x_baseline[num, 1]])
    point_baseline.set_3d_properties([x_baseline[num, 2]])

    line_HBO.set_data(x_HBO[: num + 1, 0], x_HBO[: num + 1, 1])
    line_HBO.set_3d_properties(x_HBO[: num + 1, 2])

    point_HBO.set_data([x_HBO[num, 0]], [x_HBO[num, 1]])
    point_HBO.set_3d_properties([x_HBO[num, 2]])

    time_text.set_text(f'Time: {dt[num]:.2f}s')
    return line_baseline, point_baseline, line_HBO, point_HBO, time_text

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

# ============================= 2D plot (type 1) =============================
fig = plt.figure()
ax1 = plt.subplot(4, 1, 1)
ax1.plot(dt, x_ref[:, 0], label="x_ref")
ax1.plot(dt, x_baseline[:, 0], label="x_baseline")
ax1.plot(dt, x_HBO[:, 0], label="x_HBO")
ax1.legend(loc="upper right")
ax2 = plt.subplot(4, 1, 2)
ax2.plot(dt, x_ref[:, 1], label="y_ref")
ax2.plot(dt, x_baseline[:, 1], label="y_baseline")
ax2.plot(dt, x_HBO[:, 1], label="y_HBO")
ax2.legend(loc="upper right")
ax3 = plt.subplot(4, 1, 3)
ax3.plot(dt, x_ref[:, 2], label="z_ref")
ax3.plot(dt, x_baseline[:, 2], label="z_baseline")
ax3.plot(dt, x_HBO[:, 2], label="z_HBO")
ax3.legend(loc="upper right")
ax4 = plt.subplot(4, 1, 4)
ax4.plot(dt, x_ref[:, -1], label="yaw_ref")
ax4.plot(dt, x_yaw_baseline, label="yaw_baseline")
ax4.plot(dt, x_yaw_HBO, label="yaw_HBO")
ax4.legend(loc="upper right")
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True, linestyle="--", alpha=0.6)
plt.show()

# ============================= 2D plot (type 2) =============================
fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(hspace=0.2, wspace=0.2, right=0.9)
ax1 = plt.subplot(2, 2, 1)
ax1.plot(dt, x_ref[:, 0], "--", color="red", label="$x_{d}$")
ax1.plot(dt, x_baseline[:, 0], color="red", label="$x$")
ax1.plot(dt, x_ref[:, 1], "--", color="green", label="$y_{d}$")
ax1.plot(dt, x_baseline[:, 1], color="green", label="$y$")
ax1.plot(dt, x_ref[:, 2], "--", color="blue", label="$z_{d}$")
ax1.plot(dt, x_baseline[:, 2], color="blue", label="$z$")
ax1.set_title("BO-PID")
ax1.set_ylabel("Position (m)")
ax2 = plt.subplot(2, 2, 2)
ax2.plot(dt, x_ref[:, 0], "--", color="red", label="$x_{d}$")
ax2.plot(dt, x_HBO[:, 0], color="red", label="$x$")
ax2.plot(dt, x_ref[:, 1], "--", color="green", label="$y_{d}$")
ax2.plot(dt, x_HBO[:, 1], color="green", label="$y$")
ax2.plot(dt, x_ref[:, 2], "--", color="blue", label="$z_{d}$")
ax2.plot(dt, x_HBO[:, 2], color="blue", label="$z$")
ax2.set_title("HBO-PID")
ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
ax3 = plt.subplot(2, 2, 3)
ax3.plot(dt, np.degrees(x_yaw_baseline), label="$\psi$")
ax3.plot(dt, np.degrees(x_ref[:, -1]), label="$\psi_{d}$")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Yaw ($^\circ$)")
ax4 = plt.subplot(2, 2, 4)
ax4.plot(dt, np.degrees(x_yaw_HBO), label="$\psi$")
ax4.plot(dt, np.degrees(x_ref[:, -1]), label="$\psi_{d}$")
ax4.set_xlabel("Time (s)")
ax4.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

ymin1, ymax1 = min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(ymin1, ymax1)
ax2.set_ylim(ymin1, ymax1)
ymin2, ymax2 = min(ax3.get_ylim()[0], ax4.get_ylim()[0]), max(ax3.get_ylim()[1], ax4.get_ylim()[1])
ax3.set_ylim(ymin2, ymax2)
ax4.set_ylim(ymin2, ymax2)

for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True, linestyle="--", alpha=0.6)

plt.show()
