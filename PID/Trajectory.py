import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import YawControlEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import animation_Trajectory, generate_target_trajectory, printPID


PID_PARAM_SETS = {
    "Manual": np.array([1, 0.77, 1, 0.77, 20, 10.5, 20, 3.324], dtype=float),
    "RS": np.array(
        [
            2.074697139425429,
            1.8002557547010822,
            3.2318102503303674,
            1.6096471309988258,
            22.124103912645428,
            12.188682900042139,
            17.760882764135705,
            2.559802781742307,
        ],
        dtype=float,
    ),
    "BO": np.array(
        [
            1.4903032285039675,
            0.6555869336657205,
            3.7072932982417224,
            1.8422600068867023,
            20.62216163513675,
            13.64999389298014,
            19.4343066034699,
            2.62139093544266,
        ],
        dtype=float,
    ),
    "HBO": np.array(
        [
            2.4072319861877944,
            1.1177837202417729,
            0.931257950917781,
            0.4007567734145976,
            15.18091718709083,
            5.013179923390205,
            27.713004830064556,
            3.1048918772723426,
        ],
        dtype=float,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="PID trajectory test runner")
    parser.add_argument("-pp", "--pid_param", choices=sorted(PID_PARAM_SETS.keys()), default="Manual", help="Preset PID parameter set.")
    parser.add_argument("--traj", type=int, choices=[0, 1, 2], default=1, help="Trajectory shape: 0 ellipse, 1 four-leaf clover, 2 spiral.")
    parser.add_argument("--length", type=int, default=5000, help="Trajectory length.")
    return parser.parse_args()


def resolve_pid_params(args):
    return PID_PARAM_SETS[args.pid_param].copy(), args.pid_param


def main():
    args = parse_args()
    shape_type = args.traj
    length = args.length
    pid_params, param_name = resolve_pid_params(args)

    env = YawControlEnv()
    env.set_pid_params(PID_params=pid_params)

    pos = []
    ang = []

    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    targets, name = generate_target_trajectory(shape_type, length)

    for i in range(length):
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()

    # 打印PID参数
    printPID(env)

    # 将 pos 和 ang 列表转换为 NumPy 数组
    pos = np.array(pos)
    ang = np.array(ang)

    # 位置误差、角度误差
    print("PID ", name, "| params:", param_name)
    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
    print("pos_error", np.mean(pos_error), np.std(pos_error))
    print("ang_error", np.mean(ang_error), np.std(ang_error))

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    px = pos[:, 0]
    py = pos[:, 1]
    pz = pos[:, 2]
    roll = ang[:, 0]
    pitch = ang[:, 1]
    yaw = ang[:, 2]

    ax.plot(px, py, pz, label="track")
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], label="target")
    ax.view_init(azim=45.0, elev=30)
    plt.legend()
    plt.show()

    index = np.array(range(length)) * 0.01
    zeros = np.zeros_like(index)
    plt.subplot(3, 2, 1)
    plt.plot(index, px, label="x")
    plt.plot(index, targets[:, 0], label="x_target")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(index, pitch, label="pitch")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(index, py, label="y")
    plt.plot(index, targets[:, 1], label="y_target")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(index, roll, label="roll")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(index, pz, label="z")
    plt.plot(index, targets[:, 2], label="z_target")
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(index, yaw, label="yaw")
    plt.plot(index, targets[:, 3], label="yaw_target")
    plt.plot(index, zeros)
    plt.legend()

    plt.show()

    # 动画
    animation_Trajectory(
        t_all=np.array(range(length)) * 0.01,
        dt=0.01,
        x_list=pos[:, 0],
        y_list=pos[:, 1],
        z_list=pos[:, 2],
        x_traget=targets[:, 0],
        y_traget=targets[:, 1],
        z_traget=targets[:, 2],
    )


if __name__ == "__main__":
    main()
