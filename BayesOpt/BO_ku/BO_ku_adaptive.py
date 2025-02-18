import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from EnvUAV.env_BO import YawControlEnv
from utils import generate_target_trajectory
from bayes_opt import BayesianOptimization
from utils import animation_Trajectory, printPID
import matplotlib.pyplot as plt
import copy
import pybullet as p

ang_weight = 0.3
init_PD_params = [
    2.4072319861877944,
    1.1177837202417729,
    0.931257950917781,
    0.4007567734145976,
    15.18091718709083,
    5.013179923390205,
    27.713004830064556,
    3.1048918772723426,
]


def simulate_n_steps(Px, Dx, Py, Dy, Pz, Dz, Pa, Da, env, current_target, env_state):
    env.set_state(env_state)  # 恢复环境状态

    env.new_PD_params(PD_params=[Px, Dx, Py, Dy, Pz, Dz, Pa, Da])
    env.step(current_target)
    pos_error = np.sqrt(np.sum((env.current_pos - current_target[:3]) ** 2))
    ang_error = np.degrees(np.abs(env.current_ori[2] - current_target[3]))
    total_error = pos_error + ang_weight * ang_error

    return -total_error


if __name__ == "__main__":
    # 定义优化边界为初始参数上下浮动为 2 的区间
    bounds = {
        "Px": (max(0, init_PD_params[0] - 2), init_PD_params[0] + 2),
        "Dx": (max(0, init_PD_params[1] - 2), init_PD_params[1] + 2),
        "Py": (max(0, init_PD_params[2] - 2), init_PD_params[2] + 2),
        "Dy": (max(0, init_PD_params[3] - 2), init_PD_params[3] + 2),
        "Pz": (max(0, init_PD_params[4] - 2), init_PD_params[4] + 2),
        "Dz": (max(0, init_PD_params[5] - 2), init_PD_params[5] + 2),
        "Pa": (max(0, init_PD_params[6] - 2), init_PD_params[6] + 2),
        "Da": (max(0, init_PD_params[7] - 2), init_PD_params[7] + 2),
    }
    # bounds = {
    #     "Px": (0.0, 5.0),
    #     "Dx": (0.0, 2.0),
    #     "Py": (0.0, 5.0),
    #     "Dy": (0.0, 2.0),
    #     "Pz": (15.0, 30.0),
    #     "Dz": (5.0, 15.0),
    #     "Pa": (10.0, 30.0),
    #     "Da": (0.0, 5.0),
    # }

    # 生成目标轨迹
    shape_type_traj = 0  # 0: Circle, 1: Four-leaf clover, 2: Spiral
    length_traj = 5000  # 轨迹长度
    target_trajectory, name_traj = generate_target_trajectory(
        shape_type_traj, length_traj
    )

    # 初始化环境
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    env.new_PD_params(PD_params=init_PD_params)

    pos = []
    ang = []
    final_params_list = []

    # 开始时间步循环
    for step in range(length_traj):
        print(f"==========Step {step}:==========")
        current_target = target_trajectory[step, :]  # 当前时间步的目标点
        env.new_PD_params(PD_params=init_PD_params)  # 初始化PD参数
        
        opt_params = None
        final_params = init_PD_params
        init_error = 0.0
        opt_error = 0.0

        if step >= 300:
            # 计算初始参数下的误差
            env_state = env.get_state()
            env.step(current_target)
            init_error = np.sqrt(
                np.sum((env.current_pos - current_target[:3]) ** 2)
            ) + ang_weight * np.degrees(np.abs(env.current_ori[2] - current_target[3]))
            env.set_state(env_state)

            # 优化函数调用
            optimizer = BayesianOptimization(
                f=lambda Px, Dx, Py, Dy, Pz, Dz, Pa, Da: simulate_n_steps(
                    Px, Dx, Py, Dy, Pz, Dz, Pa, Da, env, current_target, env_state
                ),
                pbounds=bounds,
                verbose=0,
                allow_duplicate_points=True,
                random_state=42,
            )

            optimizer.maximize(init_points=2, n_iter=5)
            env.set_state(env_state)
            best_params = optimizer.max["params"]  # 最优参数
            opt_error = np.abs(optimizer.max["target"])

            # 如果优化后的参数造成的误差比之前的小，则使用优化后的参数
            print(f"opt_error: {opt_error:.4f}")
            print(f"init_error: {init_error:.4f}")
            if opt_error < init_error:
                opt_params = [
                    best_params["Px"],
                    best_params["Dx"],
                    best_params["Py"],
                    best_params["Dy"],
                    best_params["Pz"],
                    best_params["Dz"],
                    best_params["Pa"],
                    best_params["Da"],
                ]
                env.new_PD_params(PD_params=opt_params)
                print("Use optimized params")
                final_params = opt_params
            else:
                env.new_PD_params(PD_params=init_PD_params)
                print("Use initial params")

        # # 验证参数是否正确记录
        # if final_params == opt_params:
        #     print('final parmas == opt_params')
        # elif final_params == init_PD_params:
        #     print('final parmas == init_PD_params')
        # else:
        #     print('what is final parmas???')

        # 执行一步仿真
        env.step(current_target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
        final_params_list.append(final_params)

        # 计算当前误差
        current_state = list(env.current_pos) + [env.current_ori[2]]
        pos_error = np.sqrt(np.sum((env.current_pos - current_target[:3]) ** 2))
        ang_error = np.degrees(np.abs(env.current_ori[2] - current_target[3]))
        total_error = pos_error + ang_weight * ang_error
        print(
            f"Current_State: {[f'{x:.4f}' for x in current_state]}, \n"
            f"target: {[f'{x:.4f}' for x in current_target]}, \n"
            f"Pos_Error: {pos_error:.4f}, "
            f"Ang_Error: {ang_error:.4f}, "
            f"Total_Error: {total_error:.4f}"
        )

        # # 验证误差计算是否正确
        # if env.is_params_equal(init_PD_params):
        #     if not np.isclose(total_error, init_error, atol=1e-6):
        #         print("Error in calculating error")
        # else:
        #     if not np.isclose(total_error, opt_error, atol=1e-6):
        #         print("Error in calculating error")

    # 仿真结束
    env.close()
    print("total step: ", env.count_step)

    # 存储数据
    pos = np.array(pos)
    ang = np.array(ang)
    final_params_list = np.array(final_params_list)
    np.save("pos_adaptive_1_3.npy", pos)
    np.save("ang_adaptive_1_3.npy", ang)
    np.save("final_params_adaptive_1_3.npy", final_params_list)

    # 输出位置误差、角度误差
    pos_error = np.sqrt(np.sum((pos - target_trajectory[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - target_trajectory[:, 3])))
    print("pos_error", np.mean(pos_error), np.std(pos_error))
    print("ang_error", np.mean(ang_error), np.std(ang_error))

    # 3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    position = np.array(pos)
    px = position[:, 0]
    py = position[:, 1]
    pz = position[:, 2]
    attitude = np.array(ang)
    roll = attitude[:, 0]
    pitch = attitude[:, 1]
    yaw = attitude[:, 2]

    ax.plot(px, py, pz, label="track")
    ax.plot(
        target_trajectory[:, 0],
        target_trajectory[:, 1],
        target_trajectory[:, 2],
        label="target",
    )
    ax.view_init(azim=45.0, elev=30)
    plt.legend()
    plt.show()

    # 轨迹图
    index = np.array(range(length_traj)) * 0.01
    zeros = np.zeros_like(index)
    plt.subplot(4, 2, 1)
    plt.plot(index, px, label="x")
    plt.plot(index, target_trajectory[:, 0], label="x_target")
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(index, pitch, label="pitch")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 3)
    plt.plot(index, py, label="y")
    plt.plot(index, target_trajectory[:, 1], label="y_target")
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(index, roll, label="roll")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 5)
    plt.plot(index, pz, label="z")
    plt.plot(index, target_trajectory[:, 2], label="z_target")
    plt.legend()

    plt.subplot(4, 2, 6)
    plt.plot(index, yaw, label="yaw")
    plt.plot(index, target_trajectory[:, 3], label="yaw_target")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 7)
    plt.plot(index, pos_error, label="pos_error")
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(4, 2, 8)
    plt.plot(index, ang_error, label="ang_error")
    plt.plot(index, zeros)
    plt.legend()

    plt.show()

    # 动画
    animation_Trajectory(
        t_all=np.array(range(length_traj)) * 0.01,
        dt=0.01,
        x_list=pos[:, 0],
        y_list=pos[:, 1],
        z_list=pos[:, 2],
        x_traget=target_trajectory[:, 0],
        y_traget=target_trajectory[:, 1],
        z_traget=target_trajectory[:, 2],
    )
