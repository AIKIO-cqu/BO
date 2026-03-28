import argparse
import os
import sys
import time
from itertools import product
from typing import Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from EnvUAV.env import YawControlEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random/Grid search for PID parameters")
    parser.add_argument("--mode", choices=["random", "two_stage", "grid"], default="random")
    parser.add_argument(
        "--shape-type",
        type=int,
        default=None,
        help="Trajectory shape: 0 ellipse, 1 four-leaf clover, 2 spiral."
             " Default is 0 for random/grid, 2 for two_stage.",
    )
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--stage-iterations", type=int, default=50)
    parser.add_argument("--split-step", type=int, default=300)

    parser.add_argument("--ang-weight-random", type=float, default=0.3)
    parser.add_argument("--ang-weight-grid", type=float, default=0.1)
    parser.add_argument("--ang-weight-stage1", type=float, default=0.5)
    parser.add_argument("--ang-weight-stage2", type=float, default=0.3)

    parser.add_argument("--grid-topk", type=int, default=10)
    return parser.parse_args()


def generate_target_trajectory(shape_type: int, length: int) -> Tuple[np.ndarray, str]:
    if shape_type == 0:
        name = "Ellipse"
        index = np.array(range(length)) / length
        tx = 2 * np.cos(2 * np.pi * index)
        ty = 2 * np.sin(2 * np.pi * index)
        tz = -np.cos(2 * np.pi * index) - np.sin(2 * np.pi * index)
        tpsi = np.sin(2 * np.pi * index) * np.pi / 3 * 2
    elif shape_type == 1:
        name = "Four-leaf clover"
        index = np.array(range(length)) / length * 2
        tx = 2 * np.sin(2 * np.pi * index) * np.cos(np.pi * index)
        ty = 2 * np.sin(2 * np.pi * index) * np.sin(np.pi * index)
        tz = -np.sin(2 * np.pi * index) * np.cos(np.pi * index) - np.sin(
            2 * np.pi * index
        ) * np.sin(np.pi * index)
        tpsi = np.sin(4 * np.pi * index) * np.pi / 4 * 3
    elif shape_type == 2:
        name = "Spiral"
        index = np.array(range(length)) / length * 4
        radius = 2 + 0.3 * np.sin(np.pi * index)
        tx = radius * np.cos(1.5 * np.pi * index)
        ty = radius * np.sin(1.5 * np.pi * index)
        tz = 0.5 * index - 1
        tpsi = np.cos(2 * np.pi * index) * np.pi / 4
    else:
        raise ValueError("shape_type must be 0, 1, or 2")

    return np.vstack([tx, ty, tz, tpsi]).T, name


def maybe_load_eval_functions():
    try:
        from utils import test_fixed_traj, test_fixed_traj_300_4700
        return test_fixed_traj, test_fixed_traj_300_4700
    except ModuleNotFoundError as exc:
        print(f"Skip final trajectory tests because dependency is missing: {exc}")
        return None, None


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


def random_bounds_single() -> np.ndarray:
    return np.array(
        [
            [0.0, 5.0],
            [0.0, 5.0],
            [0.0, 5.0],
            [0.0, 5.0],
            [15.0, 30.0],
            [5.0, 15.0],
            [10.0, 20.0],
            [0.0, 5.0],
        ],
        dtype=float,
    )


def random_bounds_two_stage() -> np.ndarray:
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


def grid_intervals() -> list:
    param_ranges = [
        (0.0, 5.0, 2),
        (0.0, 5.0, 2),
        (0.0, 5.0, 2),
        (0.0, 5.0, 2),
        (10.0, 30.0, 4),
        (5.0, 15.0, 2),
        (10.0, 30.0, 4),
        (0.0, 5.0, 2),
    ]

    splits = []
    for low, high, num_splits in param_ranges:
        step = (high - low) / num_splits
        points = [low + i * step for i in range(num_splits)] + [high]
        intervals = list(zip(points[:-1], points[1:]))
        splits.append(intervals)

    return list(product(*splits))


def random_search(
    objective_fn,
    bounds: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
    tag: str,
) -> Tuple[np.ndarray, float]:
    best_params = None
    best_error = float("inf")

    for iteration in range(n_iterations):
        params = np.array([rng.uniform(low, high) for low, high in bounds], dtype=float)
        error = float(objective_fn(params))

        if error < best_error:
            best_error = error
            best_params = params

        print(
            f"{tag} Iteration {iteration + 1}/{n_iterations}, "
            f"Error: {error:.4f}, Best Error: {best_error:.4f}"
        )

    if best_params is None:
        raise RuntimeError("Search failed to produce parameters")
    return best_params, best_error


def run_random_mode(targets: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> None:
    bounds = random_bounds_single()
    objective_fn = lambda params: evaluate_pid_on_segment(
        targets=targets,
        pid_params=params,
        start_idx=0,
        end_idx=len(targets),
        ang_weight=args.ang_weight_random,
    )

    x_best, y_best = random_search(objective_fn, bounds, args.iterations, rng, tag="Random")

    print("=====================================")
    print(f"Best params: {x_best}, best objective: {y_best}")

    test_fixed_traj, _ = maybe_load_eval_functions()
    if test_fixed_traj is not None:
        for shape_idx in range(3):
            test_fixed_traj(x_best, shape_type=shape_idx, length=args.length)


def run_two_stage_mode(targets: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> None:
    bounds = random_bounds_two_stage()
    split_step = min(max(1, args.split_step), len(targets) - 1)

    stage1_objective = lambda params: evaluate_pid_on_segment(
        targets=targets,
        pid_params=params,
        start_idx=0,
        end_idx=split_step,
        ang_weight=args.ang_weight_stage1,
    )
    stage2_objective = lambda params, stage1: evaluate_pid_on_segment(
        targets=targets,
        pid_params=params,
        start_idx=split_step,
        end_idx=len(targets),
        ang_weight=args.ang_weight_stage2,
        warmup_pid_params=stage1,
    )

    start = time.time()
    print("Optimizing for first stage...")
    x_best_stage1, y_best_stage1 = random_search(
        stage1_objective, bounds, args.stage_iterations, rng, tag="Stage1"
    )
    print(f"Best PID parameters for first stage: {x_best_stage1}, objective: {y_best_stage1}")

    print("Optimizing for second stage...")
    x_best_stage2, y_best_stage2 = random_search(
        lambda params: stage2_objective(params, x_best_stage1),
        bounds,
        args.stage_iterations,
        rng,
        tag="Stage2",
    )
    print(f"Best PID parameters for second stage: {x_best_stage2}, objective: {y_best_stage2}")
    print(f"Total time: {time.time() - start:.2f}s")

    _, test_fixed_traj_300_4700 = maybe_load_eval_functions()
    if test_fixed_traj_300_4700 is not None:
        for shape_idx in range(3):
            test_fixed_traj_300_4700(
                x_best_stage1, x_best_stage2, length=args.length, shape_type=shape_idx
            )


def run_grid_mode(targets: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> None:
    combinations = grid_intervals()

    best_params = None
    best_error = float("inf")
    record = []

    for idx, bounds in enumerate(combinations, start=1):
        params = np.array([rng.uniform(low, high) for low, high in bounds], dtype=float)
        error = evaluate_pid_on_segment(
            targets=targets,
            pid_params=params,
            start_idx=0,
            end_idx=len(targets),
            ang_weight=args.ang_weight_grid,
        )
        record.append((bounds, float(error), params))

        if error < best_error:
            best_error = float(error)
            best_params = params

        print(f"Grid {idx}/{len(combinations)}, Error: {error:.4f}, Best Error: {best_error:.4f}")

    if best_params is None:
        raise RuntimeError("Grid search failed to produce parameters")

    print("=====================================")
    print(f"Best params: {best_params}, best objective: {best_error}")

    test_fixed_traj, _ = maybe_load_eval_functions()
    if test_fixed_traj is not None:
        for shape_idx in range(3):
            test_fixed_traj(best_params, length=args.length, shape_type=shape_idx)

    record.sort(key=lambda x: x[1])
    topk = min(args.grid_topk, len(record))
    for i in range(topk):
        b, e, _ = record[i]
        print(f"Top {i + 1}: bounds={b}, error={e}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    shape_type = args.shape_type
    if shape_type is None:
        shape_type = 2 if args.mode == "two_stage" else 0

    targets, name = generate_target_trajectory(shape_type, args.length)
    print("Optimized trajectory shape:", name, " sim_time:", 0.01 * args.length)

    if args.mode == "random":
        run_random_mode(targets, args, rng)
    elif args.mode == "two_stage":
        run_two_stage_mode(targets, args, rng)
    else:
        run_grid_mode(targets, args, rng)


if __name__ == "__main__":
    main()
