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

    optimized_params = np.array(
        [
            1.2993380265742132,
            0.48646760213982443,
            2.9670528614946456,
            0.8743848738278881,
            25.339716766430136,
            11.734870484430333,
            13.063166281282532,
            1.105887514279641,
        ]
    )

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

    # optimized_params_stage1 = np.array(
    #     [
    #         3.40142924,
    #         1.98462994,
    #         4.41707509,
    #         1.05788168,
    #         23.57341602,
    #         11.84313599,
    #         15.29567556,
    #         1.99888121,
    #     ]
    # )
    # optimized_params_stage2 = np.array(
    #     [
    #         0.79070169,
    #         0.38081727,
    #         3.87565686,
    #         0.78962307,
    #         24.88344205,
    #         10.5618085,
    #         27.33922257,
    #         0.94894394,
    #     ]
    # )
    optimized_params_stage1 = np.array(
        [
            2.39477185,
            0.86768355,
            1.43116514,
            0.51403935,
            29.00194664,
            11.0686803,
            28.47214629,
            2.73507233,
        ]
    )
    optimized_params_stage2 = np.array(
        [
            3.41909826,
            0.81908229,
            3.98678341,
            0.40321345,
            24.21908227,
            12.88687091,
            18.59123469,
            0.97949161,
        ]
    )
    test_fixed_traj_300_4700(
        optimized_params_stage1, optimized_params_stage2, length, 0
    )
    test_fixed_traj_300_4700(
        optimized_params_stage1, optimized_params_stage2, length, 1
    )
    test_fixed_traj_300_4700(
        optimized_params_stage1, optimized_params_stage2, length, 2
    )


if __name__ == "__main__":
    # main1()
    main2()
