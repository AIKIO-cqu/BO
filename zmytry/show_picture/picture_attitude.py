import matplotlib.pyplot as plt
import numpy as np

step = np.array([1, 2, 3, 4, 5, 6])

error_x = np.array(
    [
        1885.7220577788246,
        1885.1470062863418,
        1886.2579763784536,
        1892.166300496436,
        1892.166300496436,
        1892.1339277400411,
    ]
)
error_y = np.array(
    [
        1286.8436589876924,
        1283.6684186612365,
        1287.3580862086017,
        1290.0061922061439,
        1290.0061922061439,
        1289.9168189695465,
    ]
)
error_z = np.array(
    [
        311.43079588046965,
        310.41891993904915,
        310.6460347475668,
        326.77474301868256,
        326.77474301868256,
        326.77006763690366,
    ]
)
error_pitch = np.array(
    [
        114.27273868014285,
        115.59744141343808,
        114.89324197314076,
        109.16668396662267,
        109.16668396662267,
        109.19059153739282,
    ]
)
error_roll = np.array(
    [
        110.25604314986273,
        110.52974575575244,
        110.42185981929737,
        100.46498635704489,
        100.46498635704489,
        100.4505657677048,
    ]
)
error_yaw = np.array(
    [
        72.36341097785638,
        72.47125352035503,
        72.51460514246033,
        64.04901631206991,
        64.04901631206991,
        64.04721169145972,
    ]
)
error_total = np.array(
    [
        3780.8887054548486,
        3777.832785576173,
        3782.0918042695207,
        3782.627922357,
        3782.627922357,
        3782.509183343049,
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
plt.suptitle("attitude_controller optimized")

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
