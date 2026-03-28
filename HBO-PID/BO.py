import argparse
import os
import sys
import time
import warnings
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from EnvUAV.env import YawControlEnv
from utils import generate_target_trajectory, test_fixed_traj, test_fixed_traj_300_4700

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline Bayesian optimization for PID parameters")
    parser.add_argument("--mode", choices=["single", "two_stage"], default="single")
    parser.add_argument(
        "--shape-type",
        type=int,
        default=None,
        help="Trajectory shape: 0 ellipse, 1 four-leaf clover, 2 spiral."
             " Default is 0 for single mode, 2 for two_stage mode.",
    )
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--initial-samples", type=int, default=20)
    parser.add_argument("--candidate-size", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--stage-iterations", type=int, default=50)
    parser.add_argument("--split-step", type=int, default=300)
    parser.add_argument("--ang-weight-single", type=float, default=0.3)
    parser.add_argument("--ang-weight-stage1", type=float, default=0.5)
    parser.add_argument("--ang-weight-stage2", type=float, default=0.3)
    return parser.parse_args()


def default_bounds() -> np.ndarray:
    return np.array(
        [
            [0.0, 5.0],
            [0.0, 2.0],
            [0.0, 5.0],
            [0.0, 2.0],
            [15.0, 30.0],
            [5.0, 15.0],
            [10.0, 30.0],
            [0.0, 5.0],
        ],
        dtype=float,
    )


def build_gp() -> GaussianProcessRegressor:
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)


def acquisition_function_ei(x: np.ndarray, gp: GaussianProcessRegressor, y_min: float) -> np.ndarray:
    mu, sigma = gp.predict(x, return_std=True)
    sigma = sigma.ravel()
    improvement = y_min - mu.ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.zeros_like(improvement)
        valid = sigma > 0
        z[valid] = improvement[valid] / sigma[valid]
        ei = np.zeros_like(improvement)
        ei[valid] = improvement[valid] * norm.cdf(z[valid]) + sigma[valid] * norm.pdf(z[valid])
    return ei


def evaluate_pid_on_segment(
    targets: np.ndarray,
    pid_params: np.ndarray,
    start_idx: int,
    end_idx: int,
    ang_weight: float,
    warmup_pid_params: Optional[np.ndarray] = None,
) -> float:
    start_idx = max(0, start_idx)
    end_idx = min(len(targets), end_idx)
    if start_idx >= end_idx:
        return float("inf")

    env = YawControlEnv()
    try:
        env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

        if warmup_pid_params is not None and start_idx > 0:
            env.set_pid_params(PID_params=np.asarray(warmup_pid_params, dtype=float))
            for i in range(start_idx):
                env.step(targets[i, :])

        env.set_pid_params(PID_params=np.asarray(pid_params, dtype=float))

        pos = []
        ang = []
        for i in range(start_idx, end_idx):
            env.step(targets[i, :])
            pos.append(env.current_pos.tolist())
            ang.append(env.current_ori.tolist())
    finally:
        env.close()

    pos_arr = np.array(pos)
    ang_arr = np.array(ang)
    target_slice = targets[start_idx:end_idx]

    pos_error = np.sqrt(np.sum((pos_arr - target_slice[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs(ang_arr[:, 2] - target_slice[:, 3]))
    return float(np.mean(pos_error) + np.mean(ang_error) * ang_weight)


def simulate_trajectory_single(targets: np.ndarray, pid_params: np.ndarray, ang_weight: float) -> float:
    return evaluate_pid_on_segment(
        targets=targets,
        pid_params=pid_params,
        start_idx=0,
        end_idx=len(targets),
        ang_weight=ang_weight,
    )


def simulate_trajectory_first_stage(
    targets: np.ndarray,
    pid_params: np.ndarray,
    split_step: int,
    ang_weight: float,
) -> float:
    return evaluate_pid_on_segment(
        targets=targets,
        pid_params=pid_params,
        start_idx=0,
        end_idx=split_step,
        ang_weight=ang_weight,
    )


def simulate_trajectory_second_stage(
    targets: np.ndarray,
    pid_params: np.ndarray,
    stage1_params: np.ndarray,
    split_step: int,
    ang_weight: float,
) -> float:
    return evaluate_pid_on_segment(
        targets=targets,
        pid_params=pid_params,
        start_idx=split_step,
        end_idx=len(targets),
        ang_weight=ang_weight,
        warmup_pid_params=stage1_params,
    )


def run_bo(
    objective_fn: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_initial_samples: int,
    n_iterations: int,
    candidate_size: int,
    rng: np.random.Generator,
    tag: str,
) -> Tuple[np.ndarray, float]:
    n_dim = bounds.shape[0]
    x_sample = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_initial_samples, n_dim))
    y_sample = np.array([objective_fn(x) for x in x_sample], dtype=float).reshape(-1, 1)

    gp = build_gp()
    for i in range(n_iterations):
        gp.fit(x_sample, y_sample)

        x_candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(candidate_size, n_dim))
        y_min = float(np.min(y_sample))
        x_next = x_candidates[int(np.argmax(acquisition_function_ei(x_candidates, gp, y_min)))]

        y_next = float(objective_fn(x_next))
        print(f"{tag} Iteration {i}: error = {y_next}")

        x_sample = np.vstack((x_sample, x_next))
        y_sample = np.vstack((y_sample, y_next))

    best_idx = int(np.argmin(y_sample))
    return x_sample[best_idx], float(y_sample[best_idx])


def optimize_single_stage(
    targets: np.ndarray,
    bounds: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    objective_fn = lambda params: simulate_trajectory_single(
        targets=targets,
        pid_params=params,
        ang_weight=args.ang_weight_single,
    )
    return run_bo(
        objective_fn=objective_fn,
        bounds=bounds,
        n_initial_samples=args.initial_samples,
        n_iterations=args.iterations,
        candidate_size=args.candidate_size,
        rng=rng,
        tag="Single",
    )


def optimize_two_stage(
    targets: np.ndarray,
    bounds: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    split_step = min(max(1, args.split_step), len(targets) - 1)

    stage1_objective = lambda params: simulate_trajectory_first_stage(
        targets=targets,
        pid_params=params,
        split_step=split_step,
        ang_weight=args.ang_weight_stage1,
    )
    x_best_stage1, y_best_stage1 = run_bo(
        objective_fn=stage1_objective,
        bounds=bounds,
        n_initial_samples=args.initial_samples,
        n_iterations=args.stage_iterations,
        candidate_size=args.candidate_size,
        rng=rng,
        tag="Stage1",
    )

    stage2_objective = lambda params: simulate_trajectory_second_stage(
        targets=targets,
        pid_params=params,
        stage1_params=x_best_stage1,
        split_step=split_step,
        ang_weight=args.ang_weight_stage2,
    )
    x_best_stage2, y_best_stage2 = run_bo(
        objective_fn=stage2_objective,
        bounds=bounds,
        n_initial_samples=args.initial_samples,
        n_iterations=args.stage_iterations,
        candidate_size=args.candidate_size,
        rng=rng,
        tag="Stage2",
    )

    return x_best_stage1, y_best_stage1, x_best_stage2, y_best_stage2


def main() -> None:
    args = parse_args()

    if args.mode == "two_stage" and args.length < 2:
        raise ValueError("two_stage mode requires --length >= 2")

    rng = np.random.default_rng(args.seed)
    shape_type = args.shape_type
    if shape_type is None:
        shape_type = 0 if args.mode == "single" else 2

    bounds = default_bounds()
    targets, name_traj = generate_target_trajectory(shape_type, args.length)

    start_time = time.time()
    if args.mode == "single":
        x_best, y_best = optimize_single_stage(targets, bounds, args, rng)
        print("=====================================")
        print("Optimized trajectory shape:", name_traj, " sim_time:", 0.01 * args.length)
        print(f"Best params: {x_best}, best objective: {y_best}")

        for shape_idx in range(3):
            test_fixed_traj(x_best, length=args.length, shape_type=shape_idx)
    else:
        x_best_stage1, y_best_stage1, x_best_stage2, y_best_stage2 = optimize_two_stage(
            targets, bounds, args, rng
        )
        print("=====================================")
        print(f"Best PID for first stage: {x_best_stage1}, objective: {y_best_stage1}")
        print(f"Best PID for second stage: {x_best_stage2}, objective: {y_best_stage2}")
        print(f"Total time: {time.time() - start_time:.2f}s")

        for shape_idx in range(3):
            test_fixed_traj_300_4700(
                x_best_stage1, x_best_stage2, length=args.length, shape_type=shape_idx
            )


if __name__ == "__main__":
    main()
