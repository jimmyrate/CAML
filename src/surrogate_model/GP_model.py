import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

class GPProxyModel:
    """
    使用高斯过程回归作为代理模型，预测模型组合的性能指标。
    """
    def __init__(self, kernel=None, alpha=1e-6, normalize_y=True):
        """
        初始化高斯过程回归模型。

        参数：
        - kernel: 核函数，默认为 C(1.0) * RBF(1.0) + WhiteKernel()
        - alpha: 噪声级别，默认为 1e-6
        - normalize_y: 是否对目标值进行归一化，默认为 True
        """
        if kernel is None:
            kernel = C(1.0) * RBF(1.0) + WhiteKernel()
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y)
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        拟合高斯过程模型。

        参数：
        - X: 输入特征，形状为 (n_samples, n_features)
        - y: 目标值，形状为 (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.gp.fit(self.X_train, self.y_train)

    def predict(self, X):
        """
        预测新数据点的目标值及其标准差。

        参数：
        - X: 输入特征，形状为 (n_samples, n_features)

        返回：
        - y_mean: 预测的均值，形状为 (n_samples,)
        - y_std: 预测的标准差，形状为 (n_samples,)
        """
        X = np.array(X)
        y_mean, y_std = self.gp.predict(X, return_std=True)
        return y_mean, y_std

    def expected_improvement(self, X, xi=0.01):
        """
        计算期望改进（Expected Improvement, EI）采集函数的值。

        参数：
        - X: 输入特征，形状为 (n_samples, n_features)
        - xi: 探索与利用之间的权衡参数，默认为 0.01

        返回：
        - EI: 每个输入点的期望改进值，形状为 (n_samples,)
        """
        X = np.array(X)
        y_mean, y_std = self.predict(X)
        y_best = np.max(self.y_train)

        with np.errstate(divide='warn'):
            imp = y_mean - y_best - xi
            Z = imp / y_std
            from scipy.stats import norm
            ei = imp * norm.cdf(Z) + y_std * norm.pdf(Z)
            ei[y_std == 0.0] = 0.0
        return ei

    def suggest_next(self, candidate_X, xi=0.01):
        """
        在候选点中选择下一个评估点，基于最大期望改进。

        参数：
        - candidate_X: 候选输入特征，形状为 (n_candidates, n_features)
        - xi: 探索与利用之间的权衡参数，默认为 0.01

        返回：
        - next_x: 建议的下一个输入点，形状为 (n_features,)
        """
        ei = self.expected_improvement(candidate_X, xi)
        max_index = np.argmax(ei)
        return candidate_X[max_index]
