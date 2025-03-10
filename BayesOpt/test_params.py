import numpy as np
from utils import (
    test_fixed_traj,
    test_random_traj,
    test_fixed_traj_TD3,
    test_random_traj_TD3,
    generate_random_trajectory,
    test_fixed_traj_300_4700,
)


def main1():
    length = 5000
    baseline_PID_params = [1, 0.77, 1, 0.77, 20, 10.5, 20, 3.324]

    optimized_params = np.array([
        2.4072319861877944,
        1.1177837202417729,
        0.931257950917781,
        0.4007567734145976,
        15.18091718709083,
        5.013179923390205,
        27.713004830064556,
        3.1048918772723426,
    ])

    # test_fixed_traj(baseline_PID_params, length, 0)
    # test_fixed_traj(baseline_PID_params, length, 1)
    # test_fixed_traj(baseline_PID_params, length, 2)
    test_fixed_traj(optimized_params, length, 0)
    test_fixed_traj(optimized_params, length, 1)
    test_fixed_traj(optimized_params, length, 2)
    # test_fixed_traj_TD3(length, 0)
    # test_fixed_traj_TD3(length, 1)
    # test_fixed_traj_TD3(length, 2)

    # for i in range(3):
    #     traj_random, waypoints = generate_random_trajectory(
    #         5, [[-2, 2], [-2, 2], [-2, 2]], length, False
    #     )
    #     # test_random_traj(baseline_PID_params, length, traj_random)
    #     test_random_traj_TD3(length, traj_random)
    #     test_random_traj(optimized_params, length, traj_random)


def main2():
    length = 5000

    optimized_params_stage1 = np.array([
        0.96647918,
        0.78687774,
        0.79516885,
        0.51524935,
        16.36855107,
        13.62081968,
        29.28145682,
        3.8508648,
    ])
    optimized_params_stage2 = np.array([
        1.19426764,
        0.84112534,
        1.54244806,
        1.25169748,
        29.42951576,
        14.05371987,
        29.08377202,
        1.57802952,
    ])
    test_fixed_traj_300_4700(optimized_params_stage1, optimized_params_stage2,
                             length, 0)
    test_fixed_traj_300_4700(optimized_params_stage1, optimized_params_stage2,
                             length, 1)
    test_fixed_traj_300_4700(optimized_params_stage1, optimized_params_stage2,
                             length, 2)


if __name__ == "__main__":
    main1()
    # main2()
