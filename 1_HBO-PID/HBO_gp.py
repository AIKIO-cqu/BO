from EnvUAV.env import YawControlEnv
from testTools import test_fixed_traj, generate_traj
from noiseModel import ExponentialNoiseModel
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


ANG_WEIGHT = 0.3


# 定义 EI 采集函数
def EI(x, gp, y_min):
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


# 目标函数
def objective_function(traj, pid_params):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    env.set_pid_params(pid_params)
    pos = []
    ang = []
    for i in range(len(traj)):
        target = traj[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()
    pos = np.array(pos)
    ang = np.array(ang)
    pos_error = np.sqrt(np.sum((pos - traj[:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - traj[:, 3])))
    return np.mean(pos_error) + np.mean(ang_error) * ANG_WEIGHT


def optimization(bounds, traj, n_init=20, n_iter=100):
    # 初始样本集
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_init, bounds.shape[0]))
    y_sample = []
    for i in range(n_init):
        y_sample.append(objective_function(traj, X_sample[i]))
    y_sample = np.array(y_sample).reshape(-1, 1)

    # 目标函数模型：高斯过程回归
    kernel_gp = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel_gp, n_restarts_optimizer=10)

    # 噪声模型：自定义的指数噪声模型
    noise_model = ExponentialNoiseModel()

    # 优化循环
    gp.fit(X_sample, y_sample)
    for i in range(n_iter):
        # Step 1: 计算残差
        residuals = np.abs(y_sample.ravel() - gp.predict(X_sample).ravel())
        # Step 2: 拟合噪声模型
        noise_model.fit(X_sample, residuals)
        # Step 3: 预测新的噪声方差
        noise_std = noise_model.predict(X_sample).ravel()
        # Step 4: 更新高斯过程回归模型
        gp = GaussianProcessRegressor(
            kernel=kernel_gp, 
            alpha=noise_std ** 2,
            n_restarts_optimizer=10
        )
        # Step 5: 训练高斯过程
        gp.fit(X_sample, y_sample)
        # Step 6: 选择下一个采样点
        X_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (1000, bounds.shape[0]))
        X_next = X_candidates[np.argmax(EI(X_candidates, gp, np.min(y_sample)))]
        y_next = objective_function(traj, X_next)
        print("Iteration {}: error = {}".format(i + 1, y_next))
        # Step 7: 更新样本集
        X_sample = np.vstack((X_sample, X_next))
        y_sample = np.vstack((y_sample, y_next))

    # 获取最优参数
    X_best = X_sample[np.argmin(y_sample)]
    y_best = np.min(y_sample)

    return X_best, y_best


if __name__ == "__main__":
    # 优化边界
    bounds = np.array([[0.5, 2.0],   [0.3, 1.5],
                       [0.5, 2.0],   [0.3, 1.5],
                       [15.0, 28.0], [5.0, 15.0],
                       [15.0, 28.0], [2.0, 5.0]])

    # 优化轨迹 [0->Circle, 1->Four-leaf clover, 2->Spiral]
    traj_type = 0
    traj_length = 5000
    traj, name = generate_traj(traj_type, traj_length)

    # 优化执行
    X_best, y_best = optimization(bounds, traj, n_init=20, n_iter=100)

    # 优化结果
    print(f"==============Trajectory: {name}, sim_time: {0.01 * traj_length}==============")
    print(f"best_params: {X_best}, error: {y_best}")

    # 测试参数
    print("Testing optimized parameters...")
    for i in range(3):
        pos_error, ang_error = test_fixed_traj(i, traj_length, X_best)