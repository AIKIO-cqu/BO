import numpy as np
from datetime import datetime
from EnvUAV.env_BO import YawControlEnv
from utils import test_fixed_traj, generate_target_trajectory
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning

# 忽略 ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# 模拟多控制器轨迹跟踪任务
def simulate_trajectory_multi_control(targets, pid_params):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    env.new_PD_params(PD_params=pid_params)
    pos = []
    ang = []
    for i in range(len(targets)):
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()
    pos = np.array(pos)
    ang = np.array(ang)
    pos_error = np.sqrt(np.sum((pos - targets[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - targets[:, 3])))
    return np.mean(pos_error) + np.mean(ang_error) * 0.1


# 定义 EI 采集函数
def acquisition_function_EI(x, gp, y_min):
    mu, sigma = gp.predict(x, return_std=True)
    sigma = sigma.ravel()  # 转为 1 维
    improvement = y_min - mu.ravel()  # 计算改进
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.zeros_like(improvement)
        valid = sigma > 0  # 避免除零
        Z[valid] = improvement[valid] / sigma[valid]
        ei = np.zeros_like(improvement)
        ei[valid] = improvement[valid] * norm.cdf(Z[valid]) + sigma[valid] * norm.pdf(
            Z[valid]
        )
    return ei


# 定义 UCB 采集函数
def acquisition_function_UCB(x, kappa=2.0):
    x = np.array(x).reshape(-1, 1)
    mu, sigma = gp.predict(x, return_std=True)
    return mu + kappa * sigma


if __name__ == "__main__":
    # 定义优化边界
    bounds = np.array(
        [
            [0.0, 5.0],
            [0.0, 2.0],
            [0.0, 5.0],
            [0.0, 2.0],
            [15.0, 30.0],
            [5.0, 15.0],
            [10.0, 30.0],
            [0.0, 5.0],
        ]
    )

    # 目标轨迹
    shape_type_traj = 0  # 0: Circle, 1: Four-leaf clover, 2: Spiral
    length_traj = 5000  # 轨迹长度
    target_trajectory, name_traj = generate_target_trajectory(
        shape_type_traj, length_traj
    )

    # 初始化样本点
    n_initial_samples = 10
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial_samples, 8))
    y_sample = []
    for i in range(n_initial_samples):
        y_sample.append(
            simulate_trajectory_multi_control(target_trajectory, X_sample[i])
        )
    y_sample = np.array(y_sample).reshape(-1, 1)

    # 初始化噪声模型（使用5阶多项式回归模型）
    noise_model = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
    initial_noise = np.abs(y_sample - np.mean(y_sample))  # 初始噪声近似
    noise_model.fit(X_sample, initial_noise)

    # 定义高斯过程回归器
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # 优化循环
    n_iterations = 100
    for i in range(n_iterations):
        # 高斯过程拟合
        noise_alpha = noise_model.predict(X_sample).ravel() ** 2 + 1e-6  # 动态噪声模型
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=noise_alpha, n_restarts_optimizer=10
        )

        try:
            gp.fit(X_sample, y_sample)
        except np.linalg.LinAlgError:
            print("Adjusting alpha due to non-positive definite matrix...")
            noise_alpha += 1e-6  # 增加噪声项
            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=noise_alpha, n_restarts_optimizer=10
            )
            gp.fit(X_sample, y_sample)

        # 更新噪声模型
        residuals = np.abs(y_sample.ravel() - gp.predict(X_sample).ravel())
        noise_model.fit(X_sample, residuals)

        # 选择下一个采样点
        X_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (1000, 8))
        y_min = np.min(y_sample)
        X_next = X_candidates[
            np.argmax(acquisition_function_EI(X_candidates, gp, y_min))
        ]

        # 评估目标函数
        y_next = simulate_trajectory_multi_control(target_trajectory, X_next)

        print("Iteration {}: pos_error = {}".format(i, y_next))

        # 更新样本集
        X_sample = np.vstack((X_sample, X_next))
        y_sample = np.vstack((y_sample, y_next))

    # 获取最优参数
    X_best = X_sample[np.argmin(y_sample)]
    y_best = np.min(y_sample)

    # 定义存储文件路径
    file_path = "best_params_log.txt"

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 将结果存储到文件中
    with open(file_path, "a") as file:
        file.write(
            f"Time: {timestamp}, X_best: {X_best.tolist()}, pos+0.1*ang, {name_traj}\n"
        )

    print("=====================================")
    print("Optimized trajectory shape:", name_traj, " sim_time:", 0.01 * length_traj)
    print(f"最优输入: {X_best}, 最优输出: {y_best}")

    test_fixed_traj(X_best, length=length_traj, shape_type=0)
    test_fixed_traj(X_best, length=length_traj, shape_type=1)
    test_fixed_traj(X_best, length=length_traj, shape_type=2)
