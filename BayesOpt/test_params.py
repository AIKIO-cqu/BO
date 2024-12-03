import numpy as np
from utils import (
    test_fixed_traj,
    test_random_traj,
    generate_random_trajectory,
)


def test_params(optimized_params):
    length = 5000
    baseline_PID_params = [1, 0.77, 1, 0.77, 20, 10.5, 20, 3.324]

    test_fixed_traj(baseline_PID_params, length, 0)
    test_fixed_traj(optimized_params, length, 0)
    test_fixed_traj(baseline_PID_params, length, 1)
    test_fixed_traj(optimized_params, length, 1)
    test_fixed_traj(baseline_PID_params, length, 2)
    test_fixed_traj(optimized_params, length, 2)

    traj_random, waypoints = generate_random_trajectory(
        5, [[-2, 2], [-2, 2], [-2, 2]], length, False
    )
    test_random_traj(baseline_PID_params, length, traj_random)
    test_random_traj(test_params, length, traj_random)


if __name__ == "__main__":
    optimized_params = np.array([1, 0.77, 1, 0.77, 20, 10.5, 20, 3.324])
    test_params(optimized_params)
