import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

class ExponentialNoiseModel:
    def __init__(self, poly_degree=10, min_noise=1e-10):
        """
        异方差噪声模型：σ_ν(x) = z * exp(β^T ρ(x)) + ζ
        
        Args:
            poly_degree: 多项式特征的次数
            min_noise: 最小噪声水平 ζ
        """
        self.poly_degree = poly_degree
        self.min_noise = min_noise
        self.z = 1.0  # 缩放参数
        self.beta = None  # 线性模型参数
        self.poly_features = PolynomialFeatures(degree=poly_degree, include_bias=True)
        self.linear_model = LinearRegression()
        self.is_fitted = False
    
    def fit(self, X, residuals):
        """
        拟合噪声模型
        
        Args:
            X: 输入特征 (n_samples, n_features)
            residuals: 残差 |g(x) - ĝ(x)| (n_samples,)
        """
        # 确保 residuals 为正值且避免 log(0)
        residuals = np.maximum(residuals, 1e-10)
        
        # 对数变换：log(σ_ν(x) - ζ) = log(z) + β^T ρ(x)
        log_residuals = np.log(np.maximum(residuals - self.min_noise, 1e-10))
        
        # 生成多项式特征
        X_poly = self.poly_features.fit_transform(X)

        # 拟合线性回归模型
        self.linear_model.fit(X_poly, log_residuals)
        
        # 提取参数
        self.beta = self.linear_model.coef_
        log_z = self.linear_model.intercept_
        self.z = np.exp(log_z)
        
        self.is_fitted = True
    
    def predict(self, X):
        """
        预测噪声标准差
        
        Args:
            X: 输入特征 (n_samples, n_features)
            
        Returns:
            predicted_noise: 预测的噪声标准差 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
        
        # 生成多项式特征
        X_poly = self.poly_features.transform(X)

        # 计算 β^T ρ(x)
        linear_term = X_poly @ self.beta
        
        # 计算 σ_ν(x) = z * exp(β^T ρ(x)) + ζ
        noise_std = self.z * np.exp(linear_term) + self.min_noise
        
        return noise_std
    
    def get_params(self):
        """获取模型参数"""
        if not self.is_fitted:
            return None
        return {
            'z': self.z,
            'beta': self.beta,
            'min_noise': self.min_noise,
            'poly_degree': self.poly_degree
        }
