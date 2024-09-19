import matplotlib.pyplot as plt
import numpy as np

step = np.array([1, 2, 3, 4, 5, 6, 7])

error_x = np.array(
    [
        2010.915054336038,
        1994.0641510755177,
        2010.82222934931,
        1968.8644646461203,
        1968.9594241134064,
        1968.9594241134064,
        1968.9594241134064,
    ]
)
error_y = np.array(
    [
        1381.0669108761538,
        1377.1893324374357,
        1386.7387453418794,
        1375.367256274556,
        1375.5436063769425,
        1375.5436063769425,
        1375.5436063769425,
    ]
)
error_z = np.array(
    [
        342.94336647286514,
        342.43268412634757,
        342.9636033363863,
        341.0956817825184,
        341.07507089801436,
        341.07507089801436,
        341.07507089801436,
    ]
)
error_pitch = np.array(
    [
        146.18150070663555,
        135.11309395524233,
        150.6196888133037,
        125.73816047570841,
        125.55543844874677,
        125.55543844874677,
        125.55543844874677,
    ]
)
error_roll = np.array(
    [
        136.66784841959563,
        132.4652620953772,
        138.94040711589923,
        129.30614945100368,
        129.2780042818921,
        129.2780042818921,
        129.2780042818921,
    ]
)
error_yaw = np.array(
    [
        66.69612507488914,
        66.24710819560175,
        66.75004305375131,
        64.59957291701454,
        64.5709719397727,
        64.5709719397727,
        64.5709719397727,
    ]
)
error_total = np.array(
    [
        4084.4708058861775,
        4047.511631885522,
        4096.83471701053,
        4004.9712855469215,
        4004.982516058775,
        4004.982516058775,
        4004.982516058775,
    ]
)

baseline_error_x = 1993.3467554367285
baseline_error_y = 1368.8073792470382
baseline_error_z = 342.43206085548263
baseline_error_pitch = 138.61825671133388
baseline_error_roll = 131.9903733928769
baseline_error_yaw = 65.97398584929006
baseline_error_total = 4041.1688114927497


plt.figure(figsize=(16, 9))
plt.suptitle("x_controller optimized")

plt.subplot(4, 2, 1)
plt.plot(step, error_x, label="error_x")
plt.plot(step, baseline_error_x * np.ones_like(step), label="baseline_error_x")
plt.title("error_x")
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(step, error_pitch, label="error_pitch")
plt.plot(step, baseline_error_pitch * np.ones_like(step), label="baseline_error_pitch")
plt.title("error_pitch")
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(step, error_y, label="error_y")
plt.plot(step, baseline_error_y * np.ones_like(step), label="baseline_error_y")
plt.title("error_y")
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(step, error_roll, label="error_roll")
plt.plot(step, baseline_error_roll * np.ones_like(step), label="baseline_error_roll")
plt.title("error_roll")
plt.legend()

plt.subplot(4, 2, 5)
plt.plot(step, error_z, label="error_z")
plt.plot(step, baseline_error_z * np.ones_like(step), label="baseline_error_z")
plt.title("error_z")
plt.legend()

plt.subplot(4, 2, 6)
plt.plot(step, error_yaw, label="error_yaw")
plt.plot(step, baseline_error_yaw * np.ones_like(step), label="baseline_error_yaw")
plt.title("error_yaw")
plt.legend()

plt.subplot(4, 2, 7)
plt.plot(step, error_total, label="error_total")
plt.plot(step, baseline_error_total * np.ones_like(step), label="baseline_error_total")
plt.title("error_total")
plt.legend()

plt.tight_layout()  # 自动调整子图之间的间距
plt.show()
