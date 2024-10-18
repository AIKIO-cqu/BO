import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import YawControlEnv
import os
from utils import calculate_peak, calculate_error, calculate_rise


path = os.path.dirname(os.path.realpath(__file__))
env = YawControlEnv()
env.attitude_controller.set_param(2.6541642768986122, 0.372833690242626)

target_container = []  # 保存目标值
peak_container = []  # 保存峰值误差
error_container = []  # 保存误差
rise_container = []  # 保存上升时间

# 进行3000次测试，每次生成一个随机目标并模拟无人机的轨迹
# 期望位置和偏航角分别从[-5,-1]∪[1,5]和[-5π/6,-π/6]∪[π/6,5π/6]均匀采样
for test_point in range(3000):
    target = np.random.rand(4) * 4 + 1
    for i in range(4):
        if np.random.rand() <= 0.5:  # 以 0.5 的概率将目标值取反
            target[i] *= -1
    target[3] *= np.pi / 6

    # 模拟无人机的轨迹
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    trace = np.zeros(shape=[500, 4])
    for ep_step in range(500):
        trace[ep_step, 0:3] = env.current_pos
        trace[ep_step, 3] = env.current_ori[2]
        env.step(target)

    # 计算峰值误差、误差和上升时间
    peak = [calculate_peak(trace[:, i], target[i]) for i in range(4)]
    error = [calculate_error(trace[:, i], target[i]) for i in range(4)]
    rise = [calculate_rise(trace[:, i], target[i]) for i in range(4)]

    # 打印测试结果
    print(test_point, target, error)
    target_container.append(target)
    peak_container.append(peak)
    error_container.append(error)
    rise_container.append(rise)

    # # 画图
    # index = np.array(range(500)) * 0.01
    # plt.suptitle("test_point: " + str(test_point))
    # plt.subplot(4, 1, 1)
    # plt.plot(index, trace[:, 0], label="x")
    # plt.plot(index, target[0] * np.ones(500), label="x_target")
    # plt.legend()
    # plt.subplot(4, 1, 2)
    # plt.plot(index, trace[:, 1], label="y")
    # plt.plot(index, target[1] * np.ones(500), label="y_target")
    # plt.legend()
    # plt.subplot(4, 1, 3)
    # plt.plot(index, trace[:, 2], label="z")
    # plt.plot(index, target[2] * np.ones(500), label="z_target")
    # plt.legend()
    # plt.subplot(4, 1, 4)
    # plt.plot(index, trace[:, 3], label="yaw")
    # plt.plot(index, target[3] * np.ones(500), label="yaw_target")
    # plt.legend()
    # plt.show()

error_container = (np.array(error_container) * 1).tolist()
peak_container = (np.array(peak_container) * 100).tolist()

print("PID")
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

np.save(path + "/PID_Statics.npy", statistic)

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
