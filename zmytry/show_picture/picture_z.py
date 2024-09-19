import matplotlib.pyplot as plt
import numpy as np

step = np.array([1, 2, 3, 4, 5, 6, 7])

error_x = np.array(
    [
        1993.5488727161232,
        1989.5533520351164,
        1993.7413663309028,
        1993.7645854962943,
        1992.9435425226748,
        1992.9072044976785,
        1987.5814347580506,
    ]
)
error_y = np.array(
    [
        1357.4808621348639,
        1359.2937361105596,
        1356.380072884938,
        1356.5975516492194,
        1356.8580495700683,
        1357.0151741822472,
        1360.5671719836228,
    ]
)
error_z = np.array(
    [
        279.3012931361842,
        281.0664699341438,
        279.53959718039266,
        279.4432370162187,
        279.91404853864015,
        279.85556117173036,
        281.1342892554926,
    ]
)
error_pitch = np.array(
    [
        126.4092014807588,
        125.35604540001115,
        126.68855860075533,
        126.61886571341964,
        126.59653482778967,
        126.56272635132787,
        125.12124440301909,
    ]
)
error_roll = np.array(
    [
        131.90195453267012,
        133.1406440663991,
        132.3518384108461,
        132.20374841401303,
        131.96079684942484,
        131.92575273947892,
        132.98375295399939,
    ]
)
error_yaw = np.array(
    [
        67.63087143120596,
        67.58470081137126,
        67.63323333961048,
        67.63817029550077,
        67.59888586014428,
        67.60190468073213,
        67.56304303059227,
    ]
)
error_total = np.array(
    [
        3956.273055431806,
        3955.9949483576015,
        3956.334666747445,
        3956.266158584666,
        3955.8718581687417,
        3955.868323623195,
        3954.9509363847774,
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
plt.suptitle("z_controller optimized")

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
