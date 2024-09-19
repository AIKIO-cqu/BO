import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import YawControlEnv
import os
import json
import time


def calculate_peak(x, target):
    peak = max(x) if target >= 0 else min(x)
    return (peak - target) / target


def calculate_error(x, target):
    x = np.array(x)[-50:]
    diff = np.abs(x - target)
    error = np.average(diff)
    return error


def calculate_rise(x, target):
    x = np.abs(x)
    target = np.abs(target)
    t1 = np.max(np.argwhere((x < target * 0.1)))
    t2 = np.min(np.argwhere((x > target * 0.9)))
    return (t2 - t1) * 0.01


path = os.path.dirname(os.path.realpath(__file__))

x1 = np.load(path + "/MPC_Statics1.npy")
x2 = np.load(path + "/MPC_Statics2.npy")
x3 = np.load(path + "/MPC_Statics3.npy")
x4 = np.load(path + "/MPC_Statics4.npy")

print((x1 + x2 + x3 + x4) / 4)
exit()


env = YawControlEnv()


target_container = []
peak_container = []
error_container = []
rise_container = []
for test_point in range(500):
    print(test_point)
    target = np.random.rand(4) * 4 + 1
    for i in range(4):
        if np.random.rand() <= 0.5:
            target[i] *= -1
    target[3] *= np.pi / 6

    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    trace = np.zeros(shape=[500, 4])
    for ep_step in range(500):
        trace[ep_step, 0:3] = env.current_pos
        trace[ep_step, 3] = env.current_ori[2]
        env.step(target)

    peak = [calculate_peak(trace[:, i], target[i]) for i in range(4)]
    error = [calculate_error(trace[:, i], target[i]) for i in range(4)]
    rise = [calculate_rise(trace[:, i], target[i]) for i in range(4)]

    print(test_point, target, error)
    target_container.append(target)
    peak_container.append(peak)
    error_container.append(error)
    rise_container.append(rise)

error_container = (np.array(error_container) * 1).tolist()
peak_container = (np.array(peak_container) * 100).tolist()

print("MPC")
print("error", np.mean(error_container, axis=0), np.std(error_container, axis=0))
print("rise", np.mean(rise_container, axis=0), np.std(rise_container, axis=0))
print("peak", np.mean(peak_container, axis=0), np.std(peak_container, axis=0))

statistic = [
    np.mean(error_container, axis=0).tolist(),
    np.std(error_container, axis=0).tolist(),
    np.mean(rise_container, axis=0).tolist(),
    np.std(rise_container, axis=0).tolist(),
    np.mean(peak_container, axis=0).tolist(),
    np.std(peak_container, axis=0).tolist(),
]
np.save(path + "/MPC_Statics5.npy", statistic)
#
# x = np.load(path + '/PID_Statics.npy')
# print(x)

# with open(path+'/FixPointRecord/peak.json', 'w') as f:
#     json.dump(peak_container, f)
# with open(path+'/FixPointRecord/peak.json', 'w') as f:
#     json.dump(peak_container, f)
# with open(path+'/FixPointRecord/error.json', 'w') as f:
#     json.dump(error_container, f)
# with open(path+'/FixPointRecord/rise.json', 'w') as f:
#     json.dump(rise_container, f)

# index = np.array(range(1000))*0.01
# plt.subplot(3, 1, 1)
# plt.plot(index, trace[:,0], label='x')
# plt.subplot(3, 1, 2)
# plt.plot(index, trace[:,1], label='y')
# plt.subplot(3, 1, 3)
# plt.plot(index, trace[:,2], label='z')
# plt.show()
