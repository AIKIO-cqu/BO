import matplotlib.pyplot as plt
import numpy as np

step = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

error_x = np.array(
    [
        1908.9405494623122,
        1848.1344556170377,
        1848.1344556170377,
        1870.4794009209343,
        1843.2403512178796,
        1935.3597164448524,
        1891.1844079971481,
        1873.8331789990964,
        1792.3739467888947
    ]
)
error_y = np.array(
    [
        1290.68444953083,
        1387.1074346823566,
        1387.1074346823566,
        1321.565827881858,
        1284.38813843949,
        1317.7687039379903,
        1265.0828480622083,
        1254.2207829187478,
        1265.92369436133
    ]
)
error_z = np.array(
    [
        276.1678731980223,
        604.4172501679773,
        604.4172501679773,
        325.6037301779098,
        405.0049397144844,
        273.8315786101538,
        315.84455300160386,
        300.24381004522365,
        334.4066046270297
    ]
)
error_pitch = np.array(
    [
        104.1905878505518,
        207.5377584400754,
        207.5377584400754,
        114.61094944378414,
        136.78576598859965,
        99.71643286819636,
        120.25278269550341,
        115.769136962109,
        146.85289124090744
    ]
)
error_roll = np.array(
    [
        105.51479506982187,
        139.2111931157509,
        139.2111931157509,
        107.48635313895113,
        101.38077528183916,
        97.84167996921832,
        115.9809550186138,
        114.64909081120913,
        111.0176374170081
    ]
)
error_yaw = np.array(
    [
        65.38867219110827,
        61.2364649552929,
        61.2364649552929,
        73.03860355221772,
        67.36043921078277,
        56.98243362974385,
        74.26954782141596,
        78.96053007393893,
        86.81646103420636
    ]
)
error_total = np.array(
    [
        3750.886927302646,
        4247.64455697849,
        4247.64455697849,
        3812.784865115655,
        3838.1604098530756,
        3781.500545460155,
        3782.6150945964932,
        3737.676529810325,
        3737.3912354693766
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
plt.suptitle("altogether optimized")

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
