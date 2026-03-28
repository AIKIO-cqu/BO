import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from EnvUAV.env import YawControlEnv

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class ExponentialNoiseModel:
    def __init__(self, degree: int = 5, zeta: float = 1e-6) -> None:
        self.degree = degree
        self.zeta = zeta
        self.poly = PolynomialFeatures(degree=self.degree)
        self.beta = None
        self.z = None

    def fit(self, x: np.ndarray, residuals: np.ndarray) -> None:
        residuals = np.maximum(np.abs(residuals).ravel(), self.zeta + 1e-6)
        log_residuals = np.log(residuals - self.zeta)

        x_mapped = self.poly.fit_transform(x)
        self.beta = np.linalg.lstsq(x_mapped, log_residuals, rcond=None)[0]
        self.z = np.exp(x_mapped @ self.beta).mean()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.beta is None or self.z is None:
            raise RuntimeError("Noise model must be fitted before predict().")
        x_mapped = self.poly.transform(x)
        return self.z * np.exp(x_mapped @ self.beta) + self.zeta


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
    targets = np.vstack([tx, ty, tz, tpsi]).T
    return targets, name


def maybe_load_eval_functions():
    try:
        from utils import test_fixed_traj, test_fixed_traj_300_4700
        return test_fixed_traj, test_fixed_traj_300_4700
    except ModuleNotFoundError as exc:
        print(f"Skip final trajectory tests because dependency is missing: {exc}")
        return None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heteroscedastic Bayesian optimization for PID parameters"
    )
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

    parser.add_argument(
        "--residual-source",
        choices=["gp", "poly"],
        default="gp",
        help="How residuals are computed when fitting heteroscedastic noise model.",
    )
    parser.add_argument(
        "--noise-gp-mode",
        choices=["set_alpha", "refit"],
        default="set_alpha",
        help="set_alpha: update alpha on same GP (legacy BO_heteroscedastic)."
             " refit: fit a second GP with alpha each iteration (legacy *_gp2).",
    )
    parser.add_argument(
        "--noise-degree",
        type=int,
        default=None,
        help="Polynomial degree in ExponentialNoiseModel. Default: 5(single), 10(two_stage).",
    )
    parser.add_argument("--noise-updates", type=int, default=5)

    parser.add_argument(
        "--log-file",
        type=str,
        default="best_params_log.txt",
        help="Append best single-stage result to this file. Empty string disables logging.",
    )
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


def build_gp(alpha: Union[float, np.ndarray] = 1e-10) -> GaussianProcessRegressor:
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=alpha)


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


def fit_noise_aware_gp(
    gp_base: GaussianProcessRegressor,
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    noise_model: ExponentialNoiseModel,
    residual_source: str,
    noise_gp_mode: str,
    noise_updates: int,
    poly_residual_model: Optional[LinearRegression],
) -> GaussianProcessRegressor:
    if residual_source == "poly":
        if poly_residual_model is None:
            raise ValueError("poly residual source requires a polynomial residual model")
        poly = PolynomialFeatures(degree=10, include_bias=True)

    for _ in range(noise_updates):
        if residual_source == "gp":
            y_hat = gp_base.predict(x_sample).ravel()
        else:
            x_poly = poly.fit_transform(x_sample)
            poly_residual_model.fit(x_poly, y_sample.ravel())
            y_hat = poly_residual_model.predict(x_poly).ravel()

        residuals = np.abs(y_sample.ravel() - y_hat)
        noise_model.fit(x_sample, residuals)

    noise_alpha = noise_model.predict(x_sample).ravel() ** 2 + 1e-6

    if noise_gp_mode == "set_alpha":
        gp_base.set_params(alpha=noise_alpha)
        gp_base.fit(x_sample, y_sample)
        return gp_base

    gp_with_noise = build_gp(alpha=noise_alpha)
    gp_with_noise.fit(x_sample, y_sample)
    return gp_with_noise


def run_hetero_bo(
    objective_fn: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_initial_samples: int,
    n_iterations: int,
    candidate_size: int,
    rng: np.random.Generator,
    tag: str,
    noise_degree: int,
    residual_source: str,
    noise_gp_mode: str,
    noise_updates: int,
) -> Tuple[np.ndarray, float]:
    n_dim = bounds.shape[0]
    x_sample = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_initial_samples, n_dim))
    y_sample = np.array([objective_fn(x) for x in x_sample], dtype=float).reshape(-1, 1)

    gp = build_gp()
    noise_model = ExponentialNoiseModel(degree=noise_degree)
    poly_residual_model = LinearRegression() if residual_source == "poly" else None

    for i in range(n_iterations):
        # In set_alpha mode, previous iteration may leave alpha as an array sized to
        # the old sample count. After appending new samples, reset to scalar first.
        alpha = gp.get_params(deep=False).get("alpha", 1e-10)
        if np.ndim(alpha) == 1 and len(alpha) != len(y_sample):
            gp.set_params(alpha=1e-10)
        gp.fit(x_sample, y_sample)
        gp_eval = fit_noise_aware_gp(
            gp_base=gp,
            x_sample=x_sample,
            y_sample=y_sample,
            noise_model=noise_model,
            residual_source=residual_source,
            noise_gp_mode=noise_gp_mode,
            noise_updates=noise_updates,
            poly_residual_model=poly_residual_model,
        )

        x_candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(candidate_size, n_dim))
        y_min = float(np.min(y_sample))
        x_next = x_candidates[int(np.argmax(acquisition_function_ei(x_candidates, gp_eval, y_min)))]

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
    noise_degree: int,
) -> Tuple[np.ndarray, float]:
    objective_fn = lambda params: simulate_trajectory_single(
        targets=targets,
        pid_params=params,
        ang_weight=args.ang_weight_single,
    )
    return run_hetero_bo(
        objective_fn=objective_fn,
        bounds=bounds,
        n_initial_samples=args.initial_samples,
        n_iterations=args.iterations,
        candidate_size=args.candidate_size,
        rng=rng,
        tag="Single",
        noise_degree=noise_degree,
        residual_source=args.residual_source,
        noise_gp_mode=args.noise_gp_mode,
        noise_updates=args.noise_updates,
    )


def optimize_two_stage(
    targets: np.ndarray,
    bounds: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    noise_degree: int,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    split_step = min(max(1, args.split_step), len(targets) - 1)

    stage1_objective = lambda params: simulate_trajectory_first_stage(
        targets=targets,
        pid_params=params,
        split_step=split_step,
        ang_weight=args.ang_weight_stage1,
    )
    x_best_stage1, y_best_stage1 = run_hetero_bo(
        objective_fn=stage1_objective,
        bounds=bounds,
        n_initial_samples=args.initial_samples,
        n_iterations=args.stage_iterations,
        candidate_size=args.candidate_size,
        rng=rng,
        tag="Stage1",
        noise_degree=noise_degree,
        residual_source=args.residual_source,
        noise_gp_mode=args.noise_gp_mode,
        noise_updates=args.noise_updates,
    )

    stage2_objective = lambda params: simulate_trajectory_second_stage(
        targets=targets,
        pid_params=params,
        stage1_params=x_best_stage1,
        split_step=split_step,
        ang_weight=args.ang_weight_stage2,
    )
    x_best_stage2, y_best_stage2 = run_hetero_bo(
        objective_fn=stage2_objective,
        bounds=bounds,
        n_initial_samples=args.initial_samples,
        n_iterations=args.stage_iterations,
        candidate_size=args.candidate_size,
        rng=rng,
        tag="Stage2",
        noise_degree=noise_degree,
        residual_source=args.residual_source,
        noise_gp_mode=args.noise_gp_mode,
        noise_updates=args.noise_updates,
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

    noise_degree = args.noise_degree
    if noise_degree is None:
        noise_degree = 5 if args.mode == "single" else 10

    bounds = default_bounds()
    targets, name_traj = generate_target_trajectory(shape_type, args.length)

    start_time = time.time()
    if args.mode == "single":
        x_best, y_best = optimize_single_stage(targets, bounds, args, rng, noise_degree)

        if args.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(args.log_file, "a", encoding="utf-8") as file:
                file.write(
                    f"Time: {timestamp}, X_best: {x_best.tolist()}, "
                    f"pos+{args.ang_weight_single}*ang, {name_traj}\n"
                )

        print("=====================================")
        print("Optimized trajectory shape:", name_traj, " sim_time:", 0.01 * args.length)
        print(f"Best params: {x_best}, best objective: {y_best}")

        test_fixed_traj, _ = maybe_load_eval_functions()
        if test_fixed_traj is not None:
            for shape_idx in range(3):
                test_fixed_traj(x_best, length=args.length, shape_type=shape_idx)
    else:
        x_best_stage1, y_best_stage1, x_best_stage2, y_best_stage2 = optimize_two_stage(
            targets, bounds, args, rng, noise_degree
        )

        print("=====================================")
        print(f"Best PID for first stage: {x_best_stage1}, objective: {y_best_stage1}")
        print(f"Best PID for second stage: {x_best_stage2}, objective: {y_best_stage2}")
        print(f"Total time: {time.time() - start_time:.2f}s")

        _, test_fixed_traj_300_4700 = maybe_load_eval_functions()
        if test_fixed_traj_300_4700 is not None:
            for shape_idx in range(3):
                test_fixed_traj_300_4700(
                    x_best_stage1, x_best_stage2, length=args.length, shape_type=shape_idx
                )


if __name__ == "__main__":
    main()
