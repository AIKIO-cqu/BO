from EnvUAV.env import YawControlEnv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import animation_Trajectory, printPID
from testTools import test_fixed_traj_2stage, generate_traj, test_fixed_traj


FIRST_LENGTH = 1000


def main():
    # single-stage
    for i in range(3):
        test_fixed_traj(shape_type=i)

    # two-stage
    params1 = np.array([1.467, 0.545, 1.938, 0.347, 21.899, 13.725])
    params2 = np.array([1.827, 0.306, 1.950, 0.348, 17.150, 6.742])
    for i in range(3):
        test_fixed_traj_2stage(shape_type=i, first_length=FIRST_LENGTH, params1=params1, params2=params2)


if __name__ == "__main__":
    main()
