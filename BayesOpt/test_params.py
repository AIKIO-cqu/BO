from utils import (
    test_fixed_traj,
    test_random_traj,
    generate_random_trajectory,
)

length = 5000
baseline_PID_params = [1, 0.77, 1, 0.77, 20, 10.5, 20, 3.324]
test_params = [
    4.849978265030178,
    2.4125383569563867,
    4.923646842769255,
    2.1809910163249273,
    29.59135717933664,
    14.471962069222323,
    19.728176800825214,
    3.716158562511214,
]

test_fixed_traj(baseline_PID_params, length, 0)
test_fixed_traj(test_params, length, 0)
test_fixed_traj(baseline_PID_params, length, 1)
test_fixed_traj(test_params, length, 1)
test_fixed_traj(baseline_PID_params, length, 2)
test_fixed_traj(test_params, length, 2)

traj_random, waypoints = generate_random_trajectory(
    5, [[-2, 2], [-2, 2], [-2, 2]], length, False
)
test_random_traj(baseline_PID_params, length, traj_random)
test_random_traj(test_params, length, traj_random)
