import matplotlib.pyplot as plt
import numpy as np

step = np.array([1, 2, 3, 4, 5, 6])

error_x = np.array(
    [
        2004.5620076191906,
        1995.530319579311,
        1996.814678049514,
        1998.277508206908,
        1997.5882931345343,
        1997.392609351118,
    ]
)
error_y = np.array(
    [
        1393.8936298726076,
        1366.7310788277473,
        1329.7549882386675,
        1335.477434223474,
        1332.872514815515,
        1332.124257262609,
    ]
)
error_z = np.array(
    [
        342.21655154489025,
        341.9989284313072,
        342.91873099506086,
        343.3006240680203,
        343.1108542934853,
        343.063299269611,
    ]
)
error_pitch = np.array(
    [
        165.85764560847466,
        142.98944672293345,
        138.08428693751756,
        137.8558769170378,
        137.9921399244321,
        137.98034167706442,
    ]
)
error_roll = np.array(
    [
        181.22380975929974,
        147.70251841464403,
        129.10301447003815,
        123.45671109119405,
        125.47991638885959,
        126.0695607421349,
    ]
)
error_yaw = np.array(
    [
        70.7499984979857,
        66.72197733601352,
        65.47289063478284,
        65.31631659750752,
        65.3674559554737,
        65.38191513938362,
    ]
)
error_total = np.array(
    [
        4158.5036429024485,
        4061.674269311957,
        4002.1485893255804,
        4003.684471104141,
        4002.4111745123005,
        4002.011983441921,
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
plt.suptitle("y_controller optimized")

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
