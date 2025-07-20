from EnvUAV.env import YawControlEnv
from testTools import test_fixed_traj_2stage, generate_traj
from noiseModel import ExponentialNoiseModel
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


ANG_WEIGHT_1 = 0.2
ANG_WEIGHT_2 = 0.5
FIRST_LENGTH = 300


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


# 第一阶段目标函数
def objective_function_stage1(traj, pid_params):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    env.set_pid_params(pid_params)
    pos = []
    ang = []
    for i in range(FIRST_LENGTH):
        target = traj[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()
    pos = np.array(pos)
    ang = np.array(ang)
    pos_error = np.sqrt(np.sum((pos - traj[:FIRST_LENGTH, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - traj[:FIRST_LENGTH, 3])))
    return np.mean(pos_error) + np.mean(ang_error) * ANG_WEIGHT_1


# 第二阶段目标函数
def objective_function_stage2(traj, pid_params, X_best_stage1):
    env = YawControlEnv()
    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))
    pos = []
    ang = []

    # 阶段1：前FIRST_LENGTH步使用阶段1优化后的参数X_best_stage1
    env.set_pid_params(X_best_stage1)
    for i in range(FIRST_LENGTH):
        target = traj[i, :]
        env.step(target)
    
    # 阶段2：剩余的轨迹使用阶段2优化参数pid_params
    env.set_pid_params(pid_params)
    for i in range(FIRST_LENGTH, len(traj)):
        target = traj[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())
    env.close()

    pos = np.array(pos)
    ang = np.array(ang)
    pos_error = np.sqrt(np.sum((pos - traj[FIRST_LENGTH:, :3]) ** 2, axis=1))
    ang_error = np.degrees(np.abs((ang[:, 2] - traj[FIRST_LENGTH:, 3])))
    return np.mean(pos_error) + np.mean(ang_error) * ANG_WEIGHT_2


def optimization_stage1(bounds, traj, n_init=20, n_iter=100):
    # 初始样本集
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_init, bounds.shape[0]))
    y_sample = []
    for i in range(n_init):
        y_sample.append(objective_function_stage1(traj, X_sample[i]))
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
        y_next = objective_function_stage1(traj, X_next)
        print("Iteration {}: error = {}".format(i + 1, y_next))
        # Step 7: 更新样本集
        X_sample = np.vstack((X_sample, X_next))
        y_sample = np.vstack((y_sample, y_next))

    # 获取最优参数
    X_best = X_sample[np.argmin(y_sample)]
    y_best = np.min(y_sample)

    return X_best, y_best


def optimization_stage2(bounds, traj, X_best_stage1, n_init=20, n_iter=100):
    # 初始样本集
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_init, bounds.shape[0]))
    y_sample = []
    for i in range(n_init):
        y_sample.append(objective_function_stage2(traj, X_sample[i], X_best_stage1))
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
        y_next = objective_function_stage2(traj, X_next, X_best_stage1)
        print("Iteration {}: error = {}".format(i + 1, y_next))
        # Step 7: 更新样本集
        X_sample = np.vstack((X_sample, X_next))
        y_sample = np.vstack((y_sample, y_next))

    # 获取最优参数
    X_best = X_sample[np.argmin(y_sample)]
    y_best = np.min(y_sample)

    return X_best, y_best


if __name__ == "__main__":
    # bounds
    bounds = np.array([[0.5, 2.0],   [0.3, 1.5],
                       [0.5, 2.0],   [0.3, 1.5],
                       [15.0, 28.0], [5.0, 15.0],
                       [15.0, 28.0], [2.0, 5.0]])

    # trajectory [0->Circle, 1->Four-leaf clover, 2->Spiral]
    traj_type = 2
    traj_length = 5000
    traj, name = generate_traj(traj_type, traj_length)

    # optimization
    X_best_stage1, y_best_stage1 = optimization_stage1(bounds, traj, n_init=20, n_iter=100)
    X_best_stage2, y_best_stage2 = optimization_stage2(bounds, traj, X_best_stage1, n_init=20, n_iter=100)

    # results
    print(f"==============Trajectory: {name}, sim_time: {0.01 * traj_length}==============")
    print(f"best_params_stage1: {X_best_stage1}, error: {y_best_stage1}")
    print(f"best_params_stage2: {X_best_stage2}, error: {y_best_stage2}")

    # test
    # X_best_stage1 = np.array([3.40142924,  1.98462994,  4.41707509,  1.05788168, 23.57341602, 11.84313599, 15.29567556,  1.99888121])
    # X_best_stage2 = np.array([0.79070169,  0.38081727,  3.87565686,  0.78962307, 24.88344205, 10.5618085, 27.33922257,  0.94894394])
    print("Testing optimized parameters...")
    for i in range(3):
        pos_error, ang_error = test_fixed_traj_2stage(i, traj_length, FIRST_LENGTH, X_best_stage1, X_best_stage2)
    