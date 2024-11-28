import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import YawControlEnv
import os


env = YawControlEnv()

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
    pos = []
    ang = []
    for ep_step in range(500):
        trace[ep_step, 0:3] = env.current_pos
        trace[ep_step, 3] = env.current_ori[2]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()
    pos = np.array(pos)
    ang = np.array(ang)
    pos_error = np.sqrt(np.sum((pos - trace[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - trace[:, 3])))
    print('step:', test_point, 'pos_error:', np.mean(pos_error), 'ang_error:', np.mean(ang_error))